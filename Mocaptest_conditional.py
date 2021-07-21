'''

Trains a DRMM with short mocap sequences, then generates motions autoregressively conditioned on future movement features

TODO:
- game of tag with two characters
- bootstrap in a neutral pose
- draw flag in correct depth


TODO: robot arm test, also path planning around obstacles
  - always correct samples using CCD IK
  - subdivide each bone to multiple steps to better handle collision constraints
  - samples, samples with zero joint angle gaussian prior (to show multimodality while pruning away unnecessary contortions), samples with inequality constraint
  - generate and learn random trajectories of 3 poses, linearly interpolated. Sample conditioned on target and constraints
    - joint angles possibly not needed, can just interpolate vertices and solve angles with IK (2-bone segments)

TODO:
 - augment the data by adding random constant rotation over time to the sampled sequences

TODO: clean the data

TODO: augment the data by mirroring (more turns in both directions should allow for more flexible movement)
 - needs to be done in Unity, need joint coordinate axes

TODO: test the motion matching-like data interpolation again

TODO: augment the data using motion graphs

TODO: training clip label as discrete input - demonstrate conditioning with types of movements (only walk, or also run)

TODO: draw direction and fwd targets using https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c

'''


import numpy as np
import matplotlib.pyplot as plt
import random
import os
import glob
import tensorflow as tf
from matplotlib.patches import Ellipse, Rectangle
from DRMM import dataStream,DRMM,DataIn
import MocapUtils as mocap
from matplotlib import patches
import mpl_toolkits.mplot3d.art3d as art3d

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--windowless', default=True,action='store_true')
parser.add_argument('--train', default=False,action='store_true')
parser.add_argument('--nIter', type=int, default=200000)
parser.add_argument('--nActionsPerSample', type=int, default=5)
parser.add_argument('--nVideoTakes', type=int, default=5)
parser.add_argument('--videoTakeLength', type=int, default=30*30)  #in frames, default 30 seconds assuming 30 fps
args = parser.parse_args()
nIter=args.nIter

#Training parameters
datasets=["mocapdata/laforge_locomotion.zip","mocapdata/laforge_walk_run.zip","mocapdata/laforge_locomotion_nosliding.zip"]
datasetIdx=2
averageHistory=True
if averageHistory:
    #historyFrameTimes = [3,4]
    #historyFrameTimes = [4, 5, 6]
    #historyFrameTimes = [5, 7, 8]
    historyFrameTimes = [7, 9, 10]
    #historyFrameTimes = [11, 14, 15]
else:
    historyFrameTimes=[1,4,7]
historyMaskWeights=[0.333,0.333,0.8]
nPoseVars=mocap.nPoseVars
nControlVars=mocap.nControlVars
nActionsPerSample=args.nActionsPerSample
nTargetFrames=30    #this many future frames used for extracting control features
nBatch=256         #Training minibatch size
learningRate=0.002
train=args.train



# Model parameters
nLayers = 4
nComponentsPerLayer = 512

#Runtime parameters (for dataset 1):
nTrailFrames=5
trailSpacing=20
nUsedActions=nActionsPerSample #nActionsPerSample-smoothingHorizon
liveRender=not args.windowless
#historyMaskWeight=0.333
clampLimit=3.5
prunedPercentage=99
goalDistanceThreshold=1.0
fwdMaskWeight=1.0
goalMaskWeight=0.5 if fwdMaskWeight!=0.0 else 1.0
nSamples = 256
renderVideo=True
renderInterval=1 if renderVideo else 2  #if not rendering video, skip some frames to speed things up
videoTakeLength=args.videoTakeLength
nVideoTakes=args.nVideoTakes
useHeadHeightLimit=False
videoName="mocap_layers-{}_goal-{}_fwd-{}_limit-{}_nSamples-{}_ML".format(nLayers,goalMaskWeight,fwdMaskWeight,clampLimit,nSamples)
if useHeadHeightLimit:
    videoName+="_headHeightLimit"
renderTrail=False

# Model file name
hStr="{}".format(historyFrameTimes[0])
for h in range(1,len(historyFrameTimes)):
    hStr=hStr+"_{}".format(historyFrameTimes[h])
modelFileName = "trainedmodels/mocap_conditional_hist={}_nActions={}_nTargets={}_{}x{}".format(hStr,nActionsPerSample,nTargetFrames,nLayers,nComponentsPerLayer)
if not averageHistory:
    modelFileName=modelFileName+"_noAveraging"
if datasetIdx>0:
    modelFileName=modelFileName+"_dataset={}".format(datasetIdx)
#modelFileName=modelFileName+"_ML"
if not os.path.isfile(modelFileName+".index"):
    train=True #force training if model not yet created



# Some helper variables
nHistoryPoses=historyFrameTimes[-1]      #How many conditioning history poses per DRMM sample.
normalizationPoseTime=historyFrameTimes[-1]-1
nPackedHistory=len(historyFrameTimes)    #How many conditioning history poses per DRMM sample after averaging over the history
drmmDataDim=nPoseVars*(nPackedHistory+nActionsPerSample)+nControlVars
print("DRMM data dimensionality: ", drmmDataDim)

#Helpers: pack/unpack movement sequences and control variables to DRMM sample arrays
def packForDRMM(history,actions,controlVars):
    assert(history.shape[1]==historyFrameTimes[-1])
    nSamples=history.shape[0]
    #pack the history frames by averaging temporally
    packedHistory=np.zeros([nSamples,nPackedHistory,nPoseVars])
    currFrame=0
    for i in range(len(historyFrameTimes)):
        if averageHistory:
            packedHistory[:,i]=np.mean(history[:,currFrame:historyFrameTimes[i]],axis=1)
            currFrame = historyFrameTimes[i]
        else:
            packedHistory[:, i] = history[:, historyFrameTimes[i]-1]
    #reshape and concatenate into shape [nSamples,drmmDataDim]
    return np.concatenate([packedHistory.reshape([nSamples,nPoseVars*nPackedHistory]),
                           actions.reshape([nSamples,nPoseVars*nActionsPerSample]),
                           controlVars],axis=-1)
def unpackFromDRMM(data):
    nSamples=data.shape[0]
    #construct the full history (this is an approximation, but still useful)
    packedHistory=data[:,:nPoseVars*nPackedHistory].reshape([nSamples,nPackedHistory,nPoseVars])
    history=np.zeros([nSamples,nHistoryPoses,nPoseVars])
    currFrame=0
    for i in range(len(historyFrameTimes)):
        history[:,currFrame:historyFrameTimes[i]]=packedHistory[:,i:i+1]
        currFrame=historyFrameTimes[i]
    #extract actions
    actions=data[:,nPoseVars*nPackedHistory:(nPackedHistory+nActionsPerSample)*nPoseVars].reshape([nSamples,nActionsPerSample,nPoseVars])
    #extract controls
    controlVars=data[:,(nPackedHistory+nActionsPerSample)*nPoseVars:]
    return history,actions,controlVars

#Create a TensorFlow session
sess=tf.Session()

#Define the input data type. Note: the defaults are useBoxConstraints=False, useGaussianPrior=False, and
#maxInequalities=0, which saves some memory and compute.
inputStream=dataStream("continuous",                #This example uses continuous-valued data.
                         useBoxConstraints=True,        #We use box constraints to implement the head height limit
                         shape=[None,drmmDataDim])      #The yet unknown batch size in the first value

#Create model. Note: the constructor takes in the TensorFlow Session, but the model interface is otherwise
#designed to be framework-agnostic, to prepare for upcoming TF2 and PyTorch implementations
model=DRMM(sess=sess,
                nLayers=nLayers,
                nComponentsPerLayer=nComponentsPerLayer,
                inputs=inputStream,
                initialLearningRate=learningRate)

#Initialize
tf.global_variables_initializer().run(session=sess)

#Train or load model
saver = tf.train.Saver()
if not train:
    saver.restore(sess,modelFileName)
else:
    #Load data
    trainingSequenceLength = historyFrameTimes[-1] + nTargetFrames
    dataset = mocap.MocapDataset(datasets[datasetIdx],
                                 sequenceLength=trainingSequenceLength,
                                 optimizeForSpeed=True,
                                 normalizationPoseTime=normalizationPoseTime)
    #dataset = mocap.MocapDataset("mocapdata/sprint1_subject2.bvh.csv", sequenceLength=trainingSequenceLength,optimizeForSpeed=True)

    #Train DRMM
    def getDRMMDataBatch(nBatch):
        #sample raw sequences
        data=dataset.sampleSequences(nBatch)
        #extract the history part
        history=data[:,:nHistoryPoses]
        #extract the actions part
        actions=data[:,nHistoryPoses:nHistoryPoses+nActionsPerSample]
        #extract the control features
        controls=mocap.extractControlData(data[:,nHistoryPoses:])
        return packForDRMM(history,actions,controls)
    model.init(getDRMMDataBatch(1024))  # Data-dependent initialization
    for i in range(nIter):
        info=model.train(i/nIter,getDRMMDataBatch(nBatch))
        #Print progress
        if i%100==0 or i==nIter-1:
            print("\rDRMM training iteration {}/{}, phase {:.3f} Loss {:.3f}, logp {:.3f} learning rate {:.6f}, precision {:.3f}".format(i,nIter,i/nIter,info["loss"],info["logp"],info["lr"],info["rho"]),end="")

    #Save everything
    if not os.path.exists('trainedmodels'):
        os.makedirs('trainedmodels')
    saver.save(sess,modelFileName)

fig = plt.figure(figsize=[8.0, 4.5], tight_layout=True)
ax = fig.add_subplot(111, projection='3d')
mocap.setupAxes(ax)


#Visualize

#Test model: generate sequences of random movements autoregressively
for takeIdx in range(nVideoTakes):
    if renderVideo:
        for f in glob.glob("imagetemp/*.png"):
            os.remove(f)

    #This array keeps track of the current motion history. We bootstrap by sampling form the model
    history,_,_=unpackFromDRMM(model.sample(nSamples=1))
    history=history[0]  #we only need the first sample in the batch
    renderIdx=0
    goal=np.array([2.0,0,0])
    poseDiffs=[]
    nActionDecisions=videoTakeLength//nUsedActions
    trailFrames=[]
    previousSample=None
    for decisionIdx in range(nActionDecisions):
        #Normalize the history so that current pose in the origin, similar to the training data
        pos,R=mocap.getNormalizationPosAndR(history[normalizationPoseTime])
        localHistory=mocap.transformMotionSequence(history,pos,R)

        #Set control vars
        controlVars=np.zeros([nSamples,nControlVars])
        currPos=mocap.get_joint_position("hips",history[-1]).copy()
        currPos[2]=0 #only care about horizontal position
        goalDistance=np.linalg.norm(goal-currPos)
        if goalDistance<goalDistanceThreshold:
            #If goal reached, randomize a new goal
            if useHeadHeightLimit:
                #When testing and visualizing the head height limit, the character walks back and forth along the x-axis
                #so that we can use a side perspective
                if currPos[0]>0:
                    goal=np.array([-2.0,0,0])
                else:
                    goal = np.array([2.0, 0, 0])
            else:
                #When not testing and visualizing the head height limit, use a random goal
                goal=np.random.uniform(-4.0,4.0,size=3)
            goal[2]=0

        #clampedGoal=currPos+np.clip(goal-currPos,-1.0,1.0)
        controlVars[:,mocap.rootMeanIdx:mocap.rootMeanIdx+3]=mocap.transformMotionSequence(goal,pos,R)
        targetFwd=goal-currPos
        targetFwd/=np.linalg.norm(targetFwd)
        controlVars[:,mocap.fwdIdx:mocap.fwdIdx+3]=mocap.transformMotionSequence(targetFwd,np.zeros(3),R)

        #Pack the data for DRMM
        packed=packForDRMM(np.repeat(localHistory.reshape([1,nHistoryPoses,nPoseVars]),repeats=nSamples,axis=0),
                           np.zeros([nSamples,nActionsPerSample,nPoseVars]),
                           controlVars)

        #Determine sampling mask
        mask=np.zeros_like(packed)
        if decisionIdx>0:
            for histIdx in range(len(historyMaskWeights)):
                mask[:,histIdx*nPoseVars:(histIdx+1)*nPoseVars]=historyMaskWeights[histIdx] #mark history as known
        #for i in range(nPackedHistory):
        #    mask[:,i*nPoseVars:i*nPoseVars+3]=1.0 #mark root trajectory as known
        #mask[:,(nPackedHistory-1)*nPoseVars:nPackedHistory*nPoseVars]=1.0 #mark current pose as known
        controlStartIdx=nPoseVars*(nActionsPerSample+nPackedHistory)

        #We first sample control vars to estimate the distribution of valid control vars
        if clampLimit>0:
            samples=model.sample(inputs=DataIn(data=packed,mask=mask),sorted=False)
            _,_,sampledCtrl=unpackFromDRMM(samples)
            ctrlMean=np.mean(sampledCtrl,axis=0,keepdims=True)
            ctrlSd=np.std(sampledCtrl,axis=0,keepdims=True)
            minCtrl=ctrlMean-clampLimit*ctrlSd
            maxCtrl=ctrlMean+clampLimit*ctrlSd

            #Clamp control vars and mark them as known
            controlVars=np.clip(controlVars,minCtrl,maxCtrl)

        packed=packForDRMM(history=np.repeat(localHistory.reshape([1,nHistoryPoses,nPoseVars]),repeats=nSamples,axis=0),
                           actions=np.zeros([nSamples,nActionsPerSample,nPoseVars]),
                           controlVars=controlVars)
        mask[:,controlStartIdx+mocap.rootMeanIdx:controlStartIdx+mocap.rootMeanIdx+2]=goalMaskWeight #mark goal x,y as known
        mask[:,controlStartIdx+mocap.fwdIdx:controlStartIdx+mocap.fwdIdx+2]=fwdMaskWeight #mark fwd vector as known

        #Sample a sequence of next poses and also histories for bootstrapping
        if useHeadHeightLimit:
            #determine head height limit, decreasing linearly
            maxHeadHeight=2.0-2.0*(decisionIdx/nActionDecisions)

            #head height limit affects the maximum allowed values for action and control variables
            actionMaxValues=1000.0*np.ones([1,nActionsPerSample,nPoseVars])
            actionMaxValues[:,:,mocap.get_joint_index("head")+1]=maxHeadHeight
            controlMaxValues=1000.0*np.ones([1,nControlVars])
            controlMaxValues[:,mocap.headHeightIdx]=maxHeadHeight

            #pack min and max values similar to the other DRMM input data
            minValues = packForDRMM(
                history=-1000.0*np.ones([1,nHistoryPoses,nPoseVars]),
                actions=-1000.0*np.ones([1, nActionsPerSample, nPoseVars]),
                controlVars=-1000.0*np.ones([1,nControlVars]))
            maxValues = packForDRMM(
                history=1000.0*np.ones([1,nHistoryPoses,nPoseVars]),
                actions=actionMaxValues,
                controlVars=controlMaxValues)
            samples=model.sample(inputs=DataIn(data=packed,mask=mask,minValues=minValues,maxValues=maxValues,maxValueWeights=10.0),sorted=True)

        else:
            samples=model.sample(inputs=DataIn(data=packed,mask=mask),sorted=True)
        sampledHistory,nextPoses,_=unpackFromDRMM(samples)
        sampleIndex = 0  # pick the sample with the highest likelihood
        if previousSample is None:
             sampleIndex=0 #pick the sample with the highest likelihood
        else:
            #Pick the first sample that is not exact copy of the previous one (prevents the occasional infinite loop...)
            for sampleIndex in range(nSamples):
                if np.sum(np.abs(nextPoses[sampleIndex]-previousSample))!=0:
                    break
            if sampleIndex>0:
                print("Detected and rejected a sample that is exactly same as the previous one")
        nextPoses=nextPoses[sampleIndex]
        previousSample = nextPoses.copy()
        sampledHistory=sampledHistory[sampleIndex]

        #Convert sampled poses to global coordinates
        nextPoses=mocap.inverseTransformMotionSequence(nextPoses,pos,R)

        #On first frame, bootstrap the history from the samples
        #if decisionIdx==0:
        #    history=sampledHistory

        #Append sampled poses to the end of the history
        history[:-nUsedActions] = history[nUsedActions:].copy()
        history[-nUsedActions:]=nextPoses[:nUsedActions]

        #Render the sampled poses
        plt.figure(1)
        for i in range(nUsedActions):
            if renderTrail:
                if renderIdx % trailSpacing == 0:
                    trailFrames.append(nextPoses[i].copy())
                    if len(trailFrames)>nTrailFrames:
                        trailFrames.pop(0)
            if renderIdx % renderInterval==0:
                mocap.claWithoutLimitReset(ax)
                mocap.setupAxes(ax)
                if renderTrail:
                    for tr in trailFrames:
                        mocap.drawSkeleton(ax,
                                           tr,
                                           color="blue",
                                           alpha=0.5,
                                           updateLimits=True)

                mocap.drawSkeleton(ax,
                                   nextPoses[i],
                                   color="blue",
                                   alpha=0.8,
                                   updateLimits=True)
                if not useHeadHeightLimit:
                    mocap.drawFlag(ax,goal[0],goal[1],color="gray")
                if useHeadHeightLimit:
                    #render the head height limit as a semitransparent red plane
                    span=4.0
                    for p in np.linspace(-span,span,10):
                        mocap.drawLine([p, -span,maxHeadHeight],[p,span,maxHeadHeight],color="red",alpha=0.5)
                        mocap.drawLine([-span,p,maxHeadHeight],[span,p,maxHeadHeight],color="red",alpha=0.5)

                #p = patches.Rectangle((-3.0,-3.0), 6, 6, linewidth=1, edgecolor='r', facecolor='r',alpha=0.4)
                    #ax.add_patch(p)
                    #art3d.pathpatch_2d_to_3d(p, z=maxHeadHeight, zdir="z")
                if liveRender:
                    plt.pause(0.001)
            if renderVideo:
                plt.savefig("./imagetemp/image{:04d}.png".format(renderIdx))
                print("Saved frame ",renderIdx)
            renderIdx+=1
    if renderVideo:
        takeName=videoName+"_take-{}.mp4".format(takeIdx)
        if os.path.exists(takeName):
            os.remove(takeName)
        if os.path.isfile("screencaps2mp4.bat"):
            os.system('screencaps2mp4.bat {}'.format(takeName))
        else:
            print("Video not generated because screencaps2mp4.bat utility not found. The animation frames can be found in the imagetemp folder.")






