import numpy as np
import random
import os
import glob 
import matplotlib.pyplot as pp
from matplotlib import patches
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" #disable Tensorflow GPU usage, these simple graphs run faster on CPU
import tensorflow as tf
import DRMM as DRMM
from matplotlib import rc
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--windowless', default=False,action='store_true')
parser.add_argument('--train', default=False,action='store_true')
parser.add_argument('--nIter', type=int, default=500000)
parser.add_argument('--nSamples', type=int, default=10)
parser.add_argument('--nData', type=int, default=1000000)

args = parser.parse_args()
nIter=args.nIter

#Tex font stuff commented out
#rc('text', usetex=True)
#rc('font',**{'family':'serif','serif':['Times']})
fontsize=16
    
     
#Globals
nLayers=10
nComponentsPerLayer=256
nData=args.nData
onlineDataGeneration=False   #True is slower, but produces infinite minibatch data variety
rigType="fullbody"
useFloorIEQ=False
nBatch=256  #training minibatch size
nSamplingBatch = 64  #inference minibatch size
modelFileName="./IKTest/{}_model_{}x{}".format(rigType,nComponentsPerLayer,nLayers)
train=args.train or (not os.path.isfile(modelFileName+".index"))

#plot limits
ylim=[-0.1,2.0]
xlim=[-1,1] 

#helper
def threshold(x,val):
    return tf.clip_by_value(tf.sign(x-val),0.0,1.0)

#A bone class for constructing 2D articulated characters such as a robot arm.
#geometry is of shape [nPoints,2]
#pos is in parent local coordinates, denoting the point around which the geometry rotates
#it is assumed that the initial geometry corresponds to local angle 0
class Bone:
    def __init__(self,pos,geometry,minAngle,maxAngle):
        self.localPos=pos.copy()
        if geometry is not None:
            self.localGeometry=geometry.copy()
        else:
            self.localGeometry=None
            self.geometry=None
        self.minAngle=minAngle
        self.maxAngle=maxAngle
        self.children=[]
        self.parent=None
        self.localAngle=0
    #transform point from local to global
    def l2g(self,p):
        returnAsVector=False
        if len(p.shape)==1:
            p=np.reshape(p,[1,-1])
            returnAsVector=True
        result=np.matmul(p,self.R)
        result+=self.pos
        #if self.parent is not None:
        #    result+=np.reshape(self.parent.pos,[1,-1])
        if returnAsVector:
            result=np.reshape(result,[2])
        return result
    def addChild(self,child):
        self.children.append(child)
        child.parent=self    
    def updateGlobal(self):
        #update angle
        self.angle=self.localAngle
        if self.parent is not None:
            self.angle+=self.parent.angle

        #rotation matrix
        self.R=np.array([[np.cos(self.angle),-np.sin(self.angle)],
                            [np.sin(self.angle),np.cos(self.angle)]])

        #update origin
        self.pos=self.parent.l2g(self.localPos) if self.parent is not None else self.localPos.copy()

        #update drawn geometry
        if self.localGeometry is not None:
            self.geometry=self.l2g(self.localGeometry)

#Rig is a collection of connected bones that can be posed using the setParams method
class Rig:
    def __init__(self,rootMin,rootMax):
        self.bones=[]
        self.nParams=2  #root coordinates
        self.rootMin=rootMin
        self.rootMax=rootMax
    def addBone(self,parent,bone):
        if parent is not None:
            parent.addChild(bone)
        self.bones.append(bone)
        self.nParams+=1
        bone.updateGlobal()
        return bone
    def getParamBounds(self):
        minVals=np.zeros(self.nParams)
        maxVals=np.zeros(self.nParams)
        minVals[:2]=self.rootMin
        maxVals[:2]=self.rootMax
        idx=2
        for bone in self.bones:
            maxVals[idx]=bone.maxAngle
            minVals[idx]=bone.minAngle
            idx+=1
        return minVals,maxVals
    def draw(self,alpha=1.0,color=None,drawEndEffectors=False):
        if color is None:
            color="gray"
        for bone in self.bones:
            if bone.geometry is not None:
                pp.plot(bone.geometry[:,0],bone.geometry[:,1],color=color,alpha=alpha)
            #elif drawEndEffectors:
            #    pp.scatter(bone.pos[0],bone.pos[1],color=color,alpha=alpha,marker='.')
    def drawCOM(self,alpha=1.0):
        com=self.getCOM()
        pp.scatter(com[0],com[1],color="black",alpha=alpha)
        
    def setParams(self,params):
        self.bones[0].localPos=params[0:2]
        for i in range(len(self.bones)):
            self.bones[i].localAngle=params[i+2]
            self.bones[i].updateGlobal()
    def getCOM(self):
        com=np.zeros(2)
        for bone in self.bones:
            if bone.geometry is not None:
                com+=np.mean(bone.geometry,axis=0)
        com/=len(self.bones)
        return com

#Helpers
def box(x0,y0,x1,y1):
    return np.array([[x0,y0],[x0,y1],[x1,y1],[x1,y0],[x0,y0]])
def vec2(x,y):
    return np.array([x,y])

#Build rig
if rigType=="arm":
    useCOM=False
    rig=Rig(rootMin=np.zeros(2),rootMax=np.zeros(2))
    L=0.3
    angleRange=0.5*np.pi
    bone1=rig.addBone(None,Bone(vec2(0,0),box(0,-0.05,L,0.05),-angleRange,angleRange))
    bone2=rig.addBone(bone1,Bone(vec2(L,0),box(0,-0.05,L,0.05),-angleRange,angleRange))
    bone3=rig.addBone(bone2,Bone(vec2(L,0),box(0,-0.05,L,0.05),-angleRange,angleRange))
    end=rig.addBone(bone3,Bone(vec2(L,0),None,0,0))
    endEffectors=[end]
    #dict for end effector and other global position indices
    nRigParams=rig.nParams
    indices={"end":nRigParams}
else:
    useCOM=True
    halfTorsoW=0.15
    legLen=0.35
    rig=Rig(rootMin=np.array([-0.3,0.4]),rootMax=np.array([0.3,1.0]))
    torso=rig.addBone(None,Bone(vec2(0,0),box(-halfTorsoW,0,halfTorsoW,0.25),-0.4*np.pi,0.4*np.pi))
    torsoUpper=rig.addBone(torso,Bone(vec2(0,0.25),box(-halfTorsoW,0,halfTorsoW,0.25),-0.4*np.pi,0.4*np.pi))
    head=rig.addBone(torsoUpper,Bone(vec2(0,0.25),box(-0.075,0,0.075,0.25),-0.1*np.pi,0.1*np.pi))
    headTop=rig.addBone(head,Bone(vec2(0,0.25),None,0,0))

    rUpperArm=rig.addBone(torsoUpper,Bone(vec2(halfTorsoW,0.25),box(0,0.05,0.3,-0.05),-0.5*np.pi,0.5*np.pi))
    rForeArm=rig.addBone(rUpperArm,Bone(vec2(0.3,0),box(0,0.05,0.3,-0.05),0,0.5*np.pi))
    #rHand=rig.addBone(rForeArm,Bone(vec2(0.3,0),box(0,0.05,0.1,-0.05),-0.1*np.pi,0.1*np.pi))
    rHand=rig.addBone(rForeArm,Bone(vec2(0.3,0),None,0,0))

    lUpperArm=rig.addBone(torsoUpper,Bone(vec2(-halfTorsoW,0.25),box(0,0.05,-0.3,-0.05),-0.5*np.pi,0.5*np.pi))
    lForeArm=rig.addBone(lUpperArm,Bone(vec2(-0.3,0),box(0,0.05,-0.3,-0.05),0.5*np.pi,0))
    #lHand=rig.addBone(lForeArm,Bone(vec2(-0.3,0),box(0,0.05,-0.1,-0.05),-0.1*np.pi,0.1*np.pi))
    lHand=rig.addBone(lForeArm,Bone(vec2(-0.3,0),None,0,0))

    rThigh=rig.addBone(torso,Bone(vec2(halfTorsoW,0),box(-0.05,0,0.05,-legLen),-0.1*np.pi,0.3*np.pi))
    rShin=rig.addBone(rThigh,Bone(vec2(0,-legLen),box(-0.05,0,0.05,-legLen),-0.25*np.pi,0.25*np.pi))
    rFoot=rig.addBone(rShin,Bone(vec2(0,-legLen),None,0,0))

    lThigh=rig.addBone(torso,Bone(vec2(-halfTorsoW,0),box(-0.05,0,0.05,-legLen),0.3*np.pi,-0.1*np.pi))
    lShin=rig.addBone(lThigh,Bone(vec2(0,-legLen),box(-0.05,0,0.05,-legLen),-0.25*np.pi,0.25*np.pi))
    lFoot=rig.addBone(lShin,Bone(vec2(0,-legLen),None,0,0))
    endEffectors=[rHand,lHand,rFoot,lFoot,headTop]
    #dict for end effector and other global position indices
    nRigParams=rig.nParams
    indices={"root":0,"COM":nRigParams,"rHand":nRigParams+2,"lHand":nRigParams+4,"rFoot":nRigParams+6,"lFoot":nRigParams+8,"head":nRigParams+10}


#Compute total number of variables in training data: rig articulation parameters plus end effector positions and optional COM position
nVars=nRigParams+2*len(endEffectors)
if useCOM:
    nVars+=2

#Get an observation vector (training data vector), based on rig articulation params
def observe(params):
    rig.setParams(params)
    result=np.zeros(nVars)
    #We observe the rig parameters
    result[:nRigParams]=params[:nRigParams]
    idx=nRigParams
    if useCOM:
        result[idx:idx+2]=rig.getCOM()
        idx+=2
    #We also observe the end effector positions
    for k in range(len(endEffectors)):
        result[idx:idx+2]=endEffectors[k].pos
        idx+=2
    return result

# Load or generate data, if training does not generated the data on the fly
if train and (not onlineDataGeneration):
    if not os.path.exists("IKTest"):
        os.mkdir("IKTest")
    dataFile = "./IKTest/{}_data.npy".format(rigType)
    dataValid=False
    if os.path.isfile(dataFile):
        data = np.load(dataFile)
        if data.shape[0]==nData:
            dataValid=True
    if not dataValid:
        # Sample rig parameters to generate random poses
        print("Generating data")
        minVals, maxVals = rig.getParamBounds()
        minVals = np.reshape(minVals, [1, -1])
        maxVals = np.reshape(maxVals, [1, -1])
        # print("maxVals",maxVals)
        # print("minVals",minVals)
        params = np.random.uniform(low=minVals, high=maxVals, size=[nData, nRigParams])

        # The dataset consists of the observation vectors corresponding to the parameters
        data = np.zeros([nData, nVars])
        data[:, :nRigParams] = params
        for i in range(nData):
            if i % 1000 == 0:
                print(i)
            data[i] = observe(params[i])
        np.save(dataFile, data)
    print("Total data dimensions: ", data.shape[1])

#Helper for sampling a training data batch
def getDataBatch(nBatch):
    if onlineDataGeneration:
        batch=np.zeros([nBatch,nVars])
        params=np.random.uniform(low=minVals,high=maxVals,size=[nBatch,nRigParams])
        batch[:,:nRigParams]=params
        for i in range(nBatch):
            batch[i]=observe(params[i])
        return batch
    else:
        return data[np.random.randint(data.shape[0], size=nBatch),:]

#Create the DRMM
sess=tf.Session()
inputStream=DRMM.dataStream("continuous",               #This example uses continuous-valued data.
                         shape=[None,nVars],            #The yet unknown batch size in the first value
                         useGaussianPrior=True,         #We use a Gaussian prior
                         maxInequalities=20)            #5 end-effectors, each with 2 coordinates, and both a min and max limit for each coordinate
                         #useBoxConstraints=True)        #We use box constraints
model=DRMM.DRMM(sess=sess,
                nLayers=nLayers,
                nComponentsPerLayer=nComponentsPerLayer,
                inputs=inputStream,
                initialLearningRate=0.002)
print("Total model parameters: ",model.nParameters)
saver = tf.train.Saver()

#Initialize
tf.global_variables_initializer().run(session=sess)

#Train or load model
saver = tf.train.Saver()
if not train:
    saver.restore(sess,modelFileName)
else:
    model.init(getDataBatch(1024))  # Data-dependent initialization
    for i in range(nIter):
        '''
        The train method performs a single EM step. 
        The method takes in a batch of data and a training phase variable in range 0...1.
        The latter is used for sweeping the learning rate and E-step \rho, as described in the paper.
        '''
        info=model.train(i/nIter,getDataBatch(nBatch))

        #Print progress
        if i%100==0 or i==nIter-1:
            logp=np.mean(model.getLogP(inputs=DRMM.DataIn(data=getDataBatch(1024),mask=np.ones([1024,nVars])))) #evaluate log-likelihood of a large data batch
            print("\rIteration {}/{}, phase {:.3f} Loss {:.3f}, logp {:.3f} learning rate {:.6f}".format(i,nIter,i/nIter,info["loss"],logp,info["lr"]),end="")
    if not os.path.exists('trainedmodels'):
        os.makedir('trainedmodels')
    saver.save(sess,modelFileName)



#Settings for the testing and plotting
captions=["Samples","Hand and feet targets added", "Inequality constraint for $y$","Multiple inequalities"] #,"Refinement with prior"]
nShowns=[10,10,10,10,10]
useFeetTargets=[False,True,True,False,True]
useHandTargets=[False,True,True,False,True]
refinement=[False,False,False,False,True]
headIeqWeights=[0,0,3.0,3.0,3.0]
otherIeqWeights=[0,0,0.0,1.0,1.0]
maxYThresholds=[1.2,1.2,1.2,1.2]
minYThreshold=0.0
minXThreshold=-0.5
maxXThreshold=0.5
nPlots=len(captions)

for sampleIdx in range(args.nSamples):
    pp.figure(sampleIdx+1, figsize=[nPlots * 3.5, 3.5])
    pp.clf()
    for plotIdx in range(nPlots):
        nShown=nShowns[plotIdx]
        pp.subplot(1,nPlots,1+plotIdx)
        pp.title(captions[plotIdx],fontsize=fontsize)
        queryData=np.zeros([nSamplingBatch,nVars])
        mask=np.zeros_like(queryData)

        if plotIdx==0:
            #Unconditional samples
            samples = model.sample(nSamples=nSamplingBatch)
            for k in range(1,nShown):
                rig.setParams(samples[k])
                rig.draw(alpha=0.2)
            rig.setParams(samples[0])
            rig.draw(alpha=1.0,color='#202020',drawEndEffectors=True)
        else:
            #Other plots have conditional samples.

            #Compose the conditioning data batch and known variables mask based on the desired end effector targets
            targets=[]
            #    targets=[{"name":"COM","pos":[0.0,0.5],"mask":[1.0,0.0]},
            #                {"name":"lFoot","pos":[-0.2,0],"mask":[1.0,1.0]},
            #                {"name":"rFoot","pos":[0.2,0],"mask":[1.0,1.0]}]
            if useFeetTargets[plotIdx]:
                targets.append({"name":"lFoot","pos":[-0.2,0.0],"mask":[1.0,1.0]})
                targets.append({"name":"rFoot","pos":[0.2,0],"mask":[1.0,1.0]})
            if useHandTargets[plotIdx]:
                #targets.append({"name":"lHand","pos":[-0.7,1.0],"mask":[1.0,1.0]})
                targets.append({"name":"rHand","pos":[0.8,1.0],"mask":[1.0,1.0]})
                #targets=[{"name":"COM","pos":[0.0,0.5],"mask":[1.0,0.0]},
                #            {"name":"rHand","pos":[0.5,1.5],"mask":[1.0,1.0]},
                #            {"name":"rHand","pos":[0.5,1.5],"mask":[1.0,1.0]},
                #            {"name":"lFoot","pos":[0,0],"mask":[1.0,1.0]}]
            for target in targets:
                idx=indices[target["name"]]
                queryData[:,idx:idx+2]=target["pos"]
                mask[:,idx:idx+2]=target["mask"]
                if target["mask"][0]>0 and target["mask"][1]>0:
                    pp.scatter(target["pos"][0],target["pos"][1],marker="x",color="black",s=80)
                elif target["mask"][0]>0:
                    pp.plot([target["pos"][0],target["pos"][0]],ylim,color="black")
                else:
                    pp.plot(xlim,[target["pos"][1],target["pos"][1]],color="black")

            # Define the inequalities
            ieqs = []
            ieqTargets = ["head", "rHand", "lHand", "lFoot", "rFoot"]
            maxYThreshold = maxYThresholds[plotIdx]
            # constraint for max y
            for ieqTarget in ieqTargets:
                idx = indices[ieqTarget] + 1  # y coordinate
                ieq_a = np.zeros([1, nVars])  # need to be broadcastable into batch of model input data
                ieq_a[0, idx] = -1
                ieq_b = maxYThreshold
                ieqs.append({"a": ieq_a, "b": ieq_b, "weight": headIeqWeights[plotIdx]})
            # other constraints
            otherWeight = otherIeqWeights[plotIdx]
            # min y
            for ieqTarget in ieqTargets:
                idx = indices[ieqTarget] + 1  # y coordinate
                ieq_a = np.zeros([1, nVars])  # need to be broadcastable into batch of model input data
                ieq_a[0, idx] = 1
                ieq_b = -minYThreshold
                ieqs.append({"a": ieq_a, "b": ieq_b, "weight": otherWeight})
            # min x
            for ieqTarget in ieqTargets:
                idx = indices[ieqTarget]  # x coordinate
                ieq_a = np.zeros([1, nVars])  # need to be broadcastable into batch of model input data
                ieq_a[0, idx] = 1
                ieq_b = -minXThreshold
                ieqs.append({"a": ieq_a, "b": ieq_b, "weight": otherWeight})
            # max x
            for ieqTarget in ieqTargets:
                idx = indices[ieqTarget]  # x coordinate
                ieq_a = np.zeros([1, nVars])  # need to be broadcastable into batch of model input data
                ieq_a[0, idx] = -1
                ieq_b = maxXThreshold
                ieqs.append({"a": ieq_a, "b": ieq_b, "weight": otherWeight})

            #Sample
            samples = model.sample(inputs=DRMM.DataIn(data=queryData,mask=mask,ieqs=ieqs),sorted=True)

            #Show random samples with low alpha
            for k in range(nShown):
                rig.setParams(samples[k])
                rig.draw(alpha=0.2)

            #Show the sample with highest likelihood with full alpha
            rig.setParams(samples[0])
            rig.draw(color='#202020',drawEndEffectors=True)

            #Visualize the inequalities
            if headIeqWeights[plotIdx]>0:
                pp.gca().add_patch(patches.Rectangle((-3,maxYThreshold),6,6,linewidth=1,edgecolor='r',facecolor='r',alpha=0.4))
            if otherIeqWeights[plotIdx]>0:
                pp.gca().add_patch(patches.Rectangle((-3,minYThreshold-3),6,3,linewidth=1,edgecolor='r',facecolor='r',alpha=0.4))
                pp.gca().add_patch(patches.Rectangle((maxXThreshold,-3),3,6,linewidth=1,edgecolor='r',facecolor='r',alpha=0.4))
                pp.gca().add_patch(patches.Rectangle((minXThreshold-3,-3),3,6,linewidth=1,edgecolor='r',facecolor='r',alpha=0.4))
            #if useCOM:
            #    rig.drawCOM()

        #common plot properties
        ax=pp.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        pp.xlim(-1.0,1.0)
        pp.ylim(ylim)
        if not args.windowless:
            pp.draw()
            pp.pause(0.001)
    pp.tight_layout(pad=0.2)
    pp.savefig("images/IK_{}.png".format(sampleIdx),dpi=200)
    if os.path.exists("PNGCrop.py"):
        import PNGCrop
        PNGCrop.crop("images/IK_{}.png".format(sampleIdx))
    if not args.windowless:
        pp.show()
