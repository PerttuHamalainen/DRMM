'''

This tutorial trains a hierarchical DRMM with movement trajectories (state and action sequences produced by dynamics simulation).

After training, we sample and visualize trajectories conditional on vertical positions at specific timesteps.

'''


import numpy as np
import matplotlib.pyplot as pp
import random
import os
import tensorflow as tf
import DRMM
from DRMM import DRMMBlockHierarchy,dataStream,DataIn
import time
from matplotlib import rc

#Uncomment these to use LaTeX fonts (as used for the paper)
#rc('text', usetex=True)
#rc('font',**{'family':'serif','serif':['Times'],'size':12})

#Plotting parameters
twoColumnFormat=True

#Dynamics parameters
dynamicsType="FlappyBird"  #either "FlappyBird" or "Balloon"

#Training parameters
nIter=100000            #Number of training iterations (minibatches). Set this to a higher value, e.g., 200000 for better quality results
nBatch=64              #Training minibatch size
modelFileName="trainedmodels/tutorial_sequential_{}".format(dynamicsType)
train=not os.path.isfile(modelFileName+".index")   #by default, we do not train again if saved model found. To force retraining, set train=True

#Inference parameters
nSampleBatch=256      #Sampling minibatch size
temperature=1.0

#Plotting parameters
figScale=3.0            #Size of a single subplot in the results plot
nPlotted=10             #How many trajectories to plot

#Parameters that should not be changed, as this requires changes to the model definition
stateDim=2              #Dimensionality of the state space (vertical position & velocity)
actionDim=1             #Dimensionality of the action space (vertical acceleration)
T=32                    #Trajectory/sequence length (don't change this, as the hierarchy layers are 
dataDim=stateDim        #Total dimensionality of a sequence item

#Create movement trajectory data
def getDataBatch(nBatch):
    data=np.zeros([nBatch,T,dataDim])
    y=np.random.uniform(0.1,0.9,size=nBatch) 
    vy=np.random.uniform(-0.05,0.05,size=nBatch)
    gravity=0
    for t in range(T):
        #store state and action for this timestep
        data[:,t,0]=y
        data[:,t,1]=vy

        #update vertical velocity
        if dynamicsType=="Balloon":
            #random vertical acceleration
            acceleration=np.random.normal(0,0.02,size=nBatch)
            vy+=acceleration
            #if agent hits limits, kill vertical velocity
            yMin=0
            yMax=1
            bounciness=0.5
            y=np.clip(y+vy,yMin,yMax)
            vy*=np.abs(np.sign(y-yMin))*np.abs(np.sign(y-yMax)) #this multiplier is 0 when y==yMin or y==yMax, and 1 otherwise
        elif dynamicsType=="FlappyBird":
            gravity=-0.01
            jumpSpeed=0.06
            #jump randomly if y<yMax, always jump if y<yMin
            yMin=0.05
            yMax=0.8
            jumpRandom=np.random.binomial(1,0.11,size=nBatch)
            jumpIfTooLow=np.clip(np.sign(yMin-y),0,1)
            tooHigh=np.clip(np.sign(y-yMax),0,1)
            jump=np.clip(jumpRandom+jumpIfTooLow-2.0*tooHigh,0,1)
            #If jumping, set vy to jumpSpeed, otherwise allow it to be affected by gravity
            vy=(1.0-jump)*(vy+gravity)+jump*jumpSpeed
            #Apply velocity
            y+=vy
        else:
            raise Exception("Unknown dynamics type")

    return data


#PLOT 1: visualize training data
nPlots=4
def subplot(idx):
    if twoColumnFormat:
        pp.subplot(2,nPlots//2,idx)
    else:
        pp.subplot(1,nPlots,idx)
def hideticks():
    ax=pp.gca()
    pp.setp(ax.get_xticklabels(), visible=False)
    pp.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)

def plotTrajectories(trajectories,color='b',alpha=1.0):
    if len(trajectories.shape)==2:
        trajectories=np.expand_dims(trajectories,axis=0)
    for trajectoryIndex in range(trajectories.shape[0]):
        pp.plot(np.arange(0,T),trajectories[trajectoryIndex,:,0],color=color,alpha=alpha)
if twoColumnFormat:
    pp.figure(1,figsize=[figScale*2,figScale*2],tight_layout=True)
else:
    pp.figure(1,figsize=[figScale*nPlots,figScale],tight_layout=True)
subplot(1)
plotTrajectories(getDataBatch(nPlotted))
pp.title("Training data")
pp.ylim(-0.05,1.05)
hideticks()
pp.pause(0.001)

#Init tf
tf.reset_default_graph()
sess=tf.Session()
tf.set_random_seed(int(time.time()))

'''
The hierarchical model has DRMM blocks that model sequence segments of 7 items while decreasing the modeled segment length through striding
As the modeled data grows more complex on each level of the hierarchy, we gradually increase the DRMM layer counts.
The last DRMM block models the joint distribution of all the segments.
'''
model=DRMMBlockHierarchy(sess,
                         inputs=dataStream(dataType="continuous",shape=[None,T,dataDim],useGaussianPrior=True,useBoxConstraints=True),
                         blockDefs=[
                         {"nClasses":64,"nLayers":2,"kernelSize":7,"stride":2},   #input seq. length 32, output length 16
                         {"nClasses":64,"nLayers":3,"kernelSize":7,"stride":2},   #in 16, out 8
                         ],
                         lastBlockClasses=64,
                         lastBlockLayers=4,
                         initialLearningRate=0.002)
print("Total model parameters: ",model.nParameters)

#Train or load model
saver = tf.train.Saver()
if not train:
    saver.restore(sess,modelFileName)
else:
    #Initialize
    tf.global_variables_initializer().run(session=sess)
    model.init(getDataBatch(nBatch)) #Data-driven init with a random batch

    #Optimize
    for i in range(nIter):
        info=model.train(i/nIter,getDataBatch(nBatch))        
        if i%10==0:
            print("Stage {}/{}, Iteration {}/{}, Loss {:.3f}, learning rate {:.6f}, precision {:.3f}".format(
                info["stage"],info["nStages"],
                i,nIter,
                info["loss"],
                info["lr"],
                info["rho"]),end="\r")
    if not os.path.exists('trainedmodels'):
        os.makedirs('trainedmodels')
    saver.save(sess,modelFileName)

#PLOT 2: Visualize unconditional samples
#Sample
samples=model.sample(nSampleBatch)

#Plot
subplot(2)
pp.cla()
plotTrajectories(samples[:nPlotted])
pp.title("Unconditional samples")
pp.ylim(-0.05,1.05)
hideticks()
pp.pause(0.001)


#PLOT 3: Visualize samples conditioned on waypoint y coordinates
#Waypoint data
waypointTimesteps=[0,T//2,T-1]
waypointY=[0.3,0.7,0.5]
nWaypoints=len(waypointTimesteps)

#Input data for the sampling: First initialize to zeros, then update the values corresponding to the waypoints
samplingInputData=np.zeros([nSampleBatch,T,dataDim])
samplingMask=np.zeros_like(samplingInputData)  

#Mask for the sampling: 1 for the known waypoint variables, zero otherwise
samplingInputData[:,waypointTimesteps,0]=waypointY
samplingMask[:,waypointTimesteps,0]=1.0

#Sample
samples=model.sample(inputs=DataIn(data=samplingInputData,mask=samplingMask),
                     temperature=temperature,
                     sorted=True)


#Plot
subplot(3)
plotTrajectories(samples[:nPlotted],alpha=0.3)
plotTrajectories(samples[0])
pp.scatter(waypointTimesteps,waypointY,color="gray",zorder=10)
pp.title("Samples conditioned\non waypoints")
pp.ylim(-0.05,1.05)
hideticks()



#PLOT 4: Visualize samples with start and end points and box constraints for y
#Waypoint data: start and end points
waypointTimesteps=[0,T-1]
waypointY=[0.3,0.5]
nWaypoints=len(waypointTimesteps)

#Input data for the sampling: First initialize to zeros, then update the values corresponding to the waypoints
samplingInputData=np.zeros([nSampleBatch,T,dataDim])
samplingMask=np.zeros_like(samplingInputData)  

#Mask for the sampling: 1 for the known waypoint variables, zero otherwise
samplingInputData[:,waypointTimesteps,0]=waypointY
samplingMask[:,waypointTimesteps,0]=1

#The box constraints need min and max values for every variable and timestep.
#We use these to define obstacles to avoid.
minValues=0.0*np.ones([T,dataDim])             
maxValues=1.0*np.ones([T,dataDim])
weights=np.zeros([T,dataDim])
weights[:,0]=1.0                   #set weights to 1 for y coordinates

#obstacle to pass over
minValues[7:12,0]=0.5

#obstacle to pass under
maxValues[20:25,0]=0.5


#Sample
samples=model.sample(inputs=DataIn(data=samplingInputData,
                                   mask=samplingMask,
                                   minValues=minValues,
                                   maxValues=maxValues,
                                   minValueWeights=weights,
                                   maxValueWeights=weights),
                     temperature=temperature,
                     sorted=True)

#Plot samples
subplot(4)
pp.cla()
plotTrajectories(samples[:nPlotted],alpha=0.3)
plotTrajectories(samples[0])
pp.scatter(waypointTimesteps,waypointY,color="gray",zorder=10)
pp.title("Samples conditioned on\nwaypoints and obstacles")
pp.ylim(-0.05,1.05)

#Plot the min and max values
pp.plot(np.arange(0,T),minValues[:,0],color="black")
pp.plot(np.arange(0,T),maxValues[:,0],color="black")
hideticks()

pp.savefig("images/{}.png".format(dynamicsType),dpi=200)
from utils import PNGCrop
PNGCrop.crop("images/{}.png".format(dynamicsType))
pp.show()
