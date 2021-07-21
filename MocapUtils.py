import sys, time, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zipfile
import math

# Visualization libraries
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from scipy.stats import rv_discrete


joint_list = ['hips','spine',
              'left_upper_leg', 'left_lower_leg', 'left_foot',
              'right_upper_leg', 'right_lower_leg', 'right_foot',
              'left_shoulder','left_upper_arm','left_lower_arm','left_hand','left_toes',
              'right_toes','right_shoulder', 'right_upper_arm', 'right_lower_arm', 'right_hand',
              'head','neck']
nKeyPoints=len(joint_list)
nPoseVars=nKeyPoints*3
rootDataScale=4.0

#    joint_list = ['left_hand', 'right_hand', 'left_lower_arm',
#        'right_lower_arm', 'left_upper_arm', 'right_upper_arm',
#        'left_shoulder', 'right_shoulder', 'head', 'neck', 'spine', 'hips',
#        'left_upper_leg', 'right_upper_leg', 'left_lower_leg', 'right_lower_leg',
#        'left_foot', 'right_foot', 'left_toes', 'right_toes']
connected_joints = [('hips', 'spine'), ('hips', 'left_upper_leg'),
    ('hips', 'right_upper_leg'), ('left_upper_leg', 'left_lower_leg'),
    ('left_lower_leg', 'left_foot'),  ('left_foot', 'left_toes'),
    ('right_upper_leg', 'right_lower_leg'), ('right_lower_leg', 'right_foot'),
    ('right_foot', 'right_toes'), ('spine', 'neck'), ('neck', 'head'),
    ('neck', 'left_upper_arm'), ('left_upper_arm', 'left_lower_arm'),
    ('left_lower_arm', 'left_hand'), ('neck', 'right_upper_arm'),
    ('right_upper_arm', 'right_lower_arm'), ('right_lower_arm', 'right_hand')]

DEBUG_LEVEL=1
CSV_COLUMNS = ['is_first', 'hips', 'spine', 'left_upper_leg', 'left_lower_leg',
    'left_foot', 'right_upper_leg', 'right_lower_leg', 'right_foot', 'left_shoulder',
    'left_upper_arm', 'left_lower_arm', 'left_hand', 'left_toes', 'right_toes',
    'right_shoulder', 'right_upper_arm', 'right_lower_arm', 'right_hand', 'head', 'neck', 'none']
REORDER_DATAFRAMES=False
CSV_COLUMNS_REORDERED = ['left_hand', 'right_hand', 'left_lower_arm', 'right_lower_arm',
    'left_upper_arm', 'right_upper_arm', 'left_shoulder', 'right_shoulder', 'head', 'neck',
    'spine', 'hips', 'left_upper_leg', 'right_upper_leg', 'left_lower_leg', 'right_lower_leg',
    'left_foot', 'right_foot', 'left_toes', 'right_toes']
CSV_COLUMNS_MIRRORED = ['right_hand', 'left_hand', 'right_lower_arm','left_lower_arm',
    'right_upper_arm', 'left_upper_arm', 'right_shoulder', 'left_shoulder', 'head', 'neck',
    'spine', 'hips', 'right_upper_leg', 'left_upper_leg', 'right_lower_leg', 'left_lower_leg',
    'right_foot', 'left_foot', 'right_toes', 'left_toes']
ROOT_COLUMN = 'hips'         # column label for the body part to use as root
DROP_COLUMNS = ['is_first', 'none']
SHOULDER_COLUMNS = ['left_shoulder', 'right_shoulder']
CSV_DELIMITER = ';'

VIEW_HORIZ_SIZE=8.0 #in meters
VIEW_HEIGHT=2.5


#Load a motion .csv or a .zip of multiple .csv:s as a list of numpy arrays, one array per .csv.
def loadMotions(fileName):
    if fileName[-4:] == ".zip":
        # Part1: Load all the csv files from the zip file to a dataframe
        # Load the zip file from the given path
        zf = zipfile.ZipFile(fileName)
        # Get list of files contained within the archive
        files = zf.infolist()
    else:
        zf = None
        files = [fileName]
    # Load the contents of the file into a dataframe
    clips = []
    for i, f in enumerate(files):
        print("Loading file: ", f)
        if zf is not None:
            dataFrame = pd.read_csv(zf.open(f), sep=CSV_DELIMITER, names=CSV_COLUMNS)
        else:
            dataFrame = pd.read_csv(f, sep=CSV_DELIMITER, names=CSV_COLUMNS)
        clips.append(processDataFrame(dataFrame))
    return clips



def processDataFrame(dataFrame, mirror=False):
    # Drop unused columns
    for column in DROP_COLUMNS:
        if column in dataFrame:
            dataFrame = dataFrame.drop(column, axis=1)
    if DEBUG_LEVEL > 1: print("Columns before reorder:\n{}".format(dataFrame.columns))
    if REORDER_DATAFRAMES:
        if mirror: dataFrame = dataFrame[CSV_COLUMNS_MIRRORED]
        else: dataFrame = dataFrame[CSV_COLUMNS_REORDERED]
    if DEBUG_LEVEL > 1: print("Columns after reorder:\n{}".format(dataFrame.columns))
    # Get root column index
    root_index = dataFrame.columns.get_loc(ROOT_COLUMN)*3
    # Get shoulder shoulder indices
    shoulder_indices = 3 * np.array([dataFrame.columns.get_loc(SHOULDER_COLUMNS[0]),
            dataFrame.columns.get_loc(SHOULDER_COLUMNS[1])])
    if DEBUG_LEVEL > 1: print(root_index)
    # Split all columns to six parts
    processedFrame = dataFrame.stack().str.extractall('([\d\.E-]+)').unstack([-2, -1])
    # Drop columns that contain rotations
    processedFrame.columns = np.arange(len(processedFrame.columns))
    drop_indices = np.arange(processedFrame.count(axis=1)[0])
    drop_indices = np.where(drop_indices % 6 > 2)
    drop_indices = drop_indices[0].tolist()
    processedFrame = processedFrame.drop(drop_indices, axis='columns')
    # Convert all columns to float
    processedFrame = processedFrame.astype(np.float64)
    # Convert from data frame to numpy array
    processedFrame = processedFrame.values
    if mirror:
        raise Exception("Mirroring not implemented yet")

    # Shuffle coordinates from xyz to xzy (Unity to PyPlot)
    nVars=processedFrame.shape[1]
    for i in range(0,nVars,3):
        tmp=processedFrame[:,i+1].copy()
        processedFrame[:,i+1]=processedFrame[:,i+2].copy()
        processedFrame[:,i+2]=tmp

    # Return the processed frame
    return processedFrame


#Computes rotation matrix from shoulder vectors. If input is multidimensional, assumes that the input vectors are along the last axis
def rotationFromShoulderVector(shoulder_vector:np.array):
    #convert to an array of vectors
    inputShape = shoulder_vector.shape
    shoulder_vector=np.reshape(shoulder_vector,[-1,shoulder_vector.shape[-1]])
    #compose the rotation matrices
    upAxis = np.array([[0, 0, 1]])
    fwd = np.cross(shoulder_vector, upAxis)
    rotations=np.concatenate([fwd, shoulder_vector, np.broadcast_to(upAxis,fwd.shape)],axis=1)
    #the result above has the rotation matrices flattened to vectors of length 9 => reshape to [3,3], adding a new array dimension
    outputShape=list(inputShape)
    outputShape[-1]=3
    outputShape.append(3)
    return np.reshape(rotations,outputShape)

#Extracts shoulder vector from a rotation matrix, as an inverse operation of the above
def shoulderVecFromR(R:np.array):
    return R.reshape([-1,3,3])[:,1].reshape(R.shape[:-1])

def getFwdVector(pose):
    inputShape = pose.shape
    if pose.ndim!=2:
        # Convert to an array of vectors to support all input array shapes
        pose = pose.reshape([-1, pose.shape[-1]])
    shoulder_vector = get_joint_position("left_shoulder", pose) - get_joint_position("right_shoulder", pose)
    shoulder_vector[:,2] = 0  # project the vector to the ground plane (assuming x,y as ground axes)
    shoulder_vector /= np.linalg.norm(shoulder_vector,axis=-1,keepdims=True)  # normalize to unit length
    upAxis = np.array([[0, 0, 1]])
    fwd=np.cross(shoulder_vector, upAxis)
    fwdShape=list(inputShape)
    fwdShape[-1]=3
    return fwd.reshape(fwdShape)


#Extracts root position on the ground plane, and rotation around the up axis,
#such that if one applies the inverse rotation, the character's facing direction will align with the x-axis
def getNormalizationPosAndR(pose:np.array):
    #Convert to an array of vectors to support all input array shapes
    inputShape=pose.shape
    pose=pose.reshape([-1,pose.shape[-1]])
    #Position is simply the first 3 coordinates, with vertical position zeroed
    pos=pose[:,:3].copy()
    pos[:,2]=0
    # Construct the rotation matrix that will normalize the shoulder vector rotation
    shoulder_vector = get_joint_position("left_shoulder", pose) - get_joint_position("right_shoulder", pose)
    shoulder_vector[:,2] = 0  # project the vector to the ground plane (assuming x,y as ground axes)
    shoulder_vector /= np.linalg.norm(shoulder_vector,axis=-1,keepdims=True)  # normalize to unit length
    R=rotationFromShoulderVector(shoulder_vector)
    #Reshape to according to the original input shape
    posShape=list(inputShape)
    posShape[-1]=3
    Rshape=list(inputShape)
    Rshape[-1]=3
    Rshape.append(3)
    return -pos.reshape(posShape), R.reshape(Rshape)

#Transposes each 2D matrix in an array of matrices
def transposeMultiple(matrixArr):
   if matrixArr.ndim==2:
       return np.transpose(matrixArr)
   # all axes except last 2 are kept as is
   newAxes=list(range(matrixArr.ndim-2))
   # last two are swapped
   newAxes.append(matrixArr.ndim-1)
   newAxes.append(matrixArr.ndim-2)
   result=np.transpose(matrixArr,axes=newAxes)
   return result

def multipleMatrixVectorProducts(A,x):
    x=np.expand_dims(x,-2)
    return np.sum(A*x,-1)


#Shift and rotate a pose or movement sequence(s)
#Assumes the last axis of data denotes poses, last axis of pos denotes the position, and two last axes of R denote a rotation matrix
def transformMotionSequence(data:np.array,pos:np.array,R:np.array):
    origShape=data.shape
    #convert input data to an array of 3D coordinates so that we can translate and rotate everything at once
    tempShape=list(origShape)
    tempShape[-1]=tempShape[-1]//3
    tempShape.append(3)
    data=data.reshape(tempShape).copy()
    #convert pos and R shapes accordingly
    pos=np.expand_dims(pos,-2)
    R=np.expand_dims(R,-3)
    #calculate the transform
    data += pos
    data=multipleMatrixVectorProducts(R,data)
    return data.reshape(origShape)

def inverseTransformMotionSequence(data:np.array,pos:np.array,R:np.array):
    origShape=data.shape
    #convert input data to an array of 3D coordinates so that we can translate and rotate everything at once
    tempShape=list(origShape)
    tempShape[-1]=tempShape[-1]//3
    tempShape.append(3)
    data=data.reshape(tempShape).copy()
    #convert pos and R shapes accordingly
    pos=np.expand_dims(pos,-2)
    R=np.expand_dims(R,-3)
    #print("data,pos,R shapes",data.shape,pos.shape,R.shape)
    data=multipleMatrixVectorProducts(transposeMultiple(R),data)
    data -= pos
    return data.reshape(origShape)



#Normalize one or more movement sequences so that the first poses are in origin and facing along the x axis
#assumes input shape [nSequences,sequenceLength,nCoords]
def normalizeSequences(data:np.array,normalizationPoseTime):
    pos, R = getNormalizationPosAndR(data[:,normalizationPoseTime:normalizationPoseTime+1])
    data = transformMotionSequence(data, pos, R)
    return data


def globalToLocal(data:np.array):
    pos, R = getNormalizationPosAndR(data)
    localPoses=transformMotionSequence(data,pos,R)
    return np.concatenate([rootDataScale*pos,rootDataScale*shoulderVecFromR(R),localPoses],axis=-1)

def localToGlobal(data:np.array):
    #convert to a list of poses to allow easy normalization and extracting of position and rotation
    inputShape=data.shape
    data=data.reshape([-1,data.shape[-1]])
    #extract pos & shoulder vectors, and normalize the shoulder for safety (e.g., a machine learning system might output denormalized vectors)
    pos=(1.0/rootDataScale)*data[:,:3]
    shoulderVec=data[:,3:6].copy()
    shoulderVec/=np.linalg.norm(shoulderVec,axis=1,keepdims=True)
    R=rotationFromShoulderVector(shoulderVec)
    result=inverseTransformMotionSequence(data[:,6:],pos,R)
    #reshape according to the original input shape
    outputShape=list(inputShape)
    outputShape[-1]-=6
    return result.reshape(outputShape)



from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
def setupAxes(ax):
    ax.grid(True)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.xaxis._axinfo['tick']['outward_factor'] = 0
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    #ax.set_xticks(np.linspace(-10.0,10.0,10))
    #ax.set_yticks(np.linspace(-10.0,10.0,10))
    ax.set_zticks([])
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.xaxis.set_major_locator(MultipleLocator(2)) #use if using pyplot's default grid
    ax.yaxis.set_major_locator(MultipleLocator(2))
    #ax.zaxis.set_major_locator(MultipleLocator(1))
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])


def get_joint_index(joint_name):
    return joint_list.index(joint_name) * 3


def get_joint_position(joint_name,joint_arr):
    joint_index = joint_list.index(joint_name) * 3
    ndims=len(joint_arr.shape)
    if ndims==1:
        return joint_arr[joint_index:joint_index + 3]  # [joint_index,joint_index+2,joint_index+3]]
    elif ndims==2:
        return joint_arr[:,joint_index:joint_index + 3]  # [joint_index,joint_index+2,joint_index+3]]
    elif ndims==3:
        return joint_arr[:,:,joint_index:joint_index + 3]  # [joint_index,joint_index+2,joint_index+3]]
    else:
        raise Exception("input shape not supported")

def drawSkeleton(axis,coords,color,alpha,updateLimits=True):
    #draw bones
    for i, (start_joint, end_joint) in enumerate(connected_joints):
        start_position = get_joint_position(start_joint,coords)
        end_position = get_joint_position(end_joint,coords)
        positions = np.stack((start_position, end_position))
        xs, ys, zs = positions[:,0], positions[:,1], positions[:,2]
        axis.plot(xs,ys,zs, color=color, alpha=alpha)
    #Camera control
    if updateLimits:
        characterOrigin = get_joint_position("hips",coords)
        margin=1.0
        currXlim=axis.get_xlim3d()
        currYlim=axis.get_ylim3d()
        currZlim=axis.get_zlim3d()
        cameraOrigin=np.array([np.mean(currXlim),np.mean(currYlim),np.mean(currZlim)])
        cameraMin=np.array([currXlim[0],currYlim[0],currZlim[0]])
        cameraMax=np.array([currXlim[1],currYlim[1],currZlim[1]])
        maxLimitViolation=np.clip(characterOrigin+margin-cameraMax,0,np.inf)
        minLimitViolation=np.clip(cameraMin-(characterOrigin-margin),0,np.inf)
        cameraOrigin+=maxLimitViolation-minLimitViolation
        #cameraOrigin=characterOrigin
        axis.set_xlim3d([cameraOrigin[0] - 0.5 * VIEW_HORIZ_SIZE, cameraOrigin[0] + 0.5 * VIEW_HORIZ_SIZE])
        axis.set_zlim3d([0.0, VIEW_HEIGHT])
        axis.set_ylim3d([cameraOrigin[1] - 0.5 * VIEW_HORIZ_SIZE, cameraOrigin[1] + 0.5 * VIEW_HORIZ_SIZE])

def drawFlag(axis,x,y,color):
    axis.plot(xs=[x,x,x+0.3,x],ys=[y,y,y,y],zs=[0,1.8,1.7,1.6],color=color)


#Extract control features
rootMeanIdx=0
fwdIdx=3
headHeightIdx=fwdIdx+3
#jointSdIdx=headHeightIdx+1
#rmsSpeedIdx=jointSdIdx+1
#rmsAccIdx=rmsSpeedIdx+1
nControlVars=headHeightIdx+1
def extractControlData(data):
    assert(data.ndim==3 or data.ndim==2)  #expect shape [nBatch,nSteps,nPoseVars]
    assert(data.shape[-1]==nPoseVars)
    if data.ndim==2:
        reshapeOutput=True
        data=data.reshape([1,data.shape[0],data.shape[1]])
    else:
        reshapeOutput=False
    result=np.zeros([data.shape[0],nControlVars])
    #mean root position
    result[:,rootMeanIdx:rootMeanIdx+3]=np.mean(get_joint_position("hips",data),axis=1)
    #mean fwd vector
    fwd=getFwdVector(data)
    fwd=np.mean(fwd,axis=1)
    fwd/=np.linalg.norm(fwd,axis=-1,keepdims=True)
    result[:, fwdIdx:fwdIdx + 3]=fwd
    #mean head height
    headHeight=np.mean(get_joint_position("head",data)[:,:,2],axis=1)
    result[:,headHeightIdx]=headHeight
    #reshape
    if reshapeOutput:
        result=result.reshape(nControlVars)
    return result

def drawLine(pt1,pt2,color,alpha):
    plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],[pt1[2],pt2[2]], color=color, alpha=alpha)

def visualize_csv(csvName,frameSkip=1):
    clips=loadMotions(csvName)
    fig=plt.figure(figsize=[8.0,4.5],tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    for clip in clips:
        controlHorizon=15
        for frameIdx in range(0,clip.shape[0]-controlHorizon,frameSkip):
            claWithoutLimitReset(ax)
            setupAxes(ax)
            pose=clip[frameIdx]
            drawSkeleton(ax, clip[frameIdx], color='blue', alpha=0.8)
            controlData=extractControlData(clip[frameIdx:frameIdx+controlHorizon])
            rootMean=controlData[rootMeanIdx:rootMeanIdx+3]
            curr=get_joint_position("hips",pose)
            drawLine(curr,rootMean,color="red",alpha=1.0)
            fwd=controlData[fwdIdx:fwdIdx+3]
            drawLine(rootMean,rootMean+fwd,color="green",alpha=1.0)
            headUpVec=np.array([rootMean[0],rootMean[1],controlData[headHeightIdx]])
            drawLine(rootMean,headUpVec,color="blue",alpha=1.0)
            # plt.draw()
            plt.pause(0.001)


# dataset.visualizeSequence(sequences[i])


class MocapDataset:
    def __init__(self,fileName,sequenceLength,optimizeForSpeed=True,localPoseCoordinates=False,normalizationPoseTime=0):
        self.sequenceLength=sequenceLength
        self.clips=loadMotions(fileName)
        self.nVarsPerFrame=nPoseVars
        self.optimizeForSpeed=optimizeForSpeed
        self.normalizationPoseTime=normalizationPoseTime
        if localPoseCoordinates:
            assert optimizeForSpeed #local pose parameterization not supported without the precomputation
        for clip in self.clips:
            assert(self.nVarsPerFrame==clip.shape[1])
        self.clipLengths=np.zeros(len(self.clips),dtype=int)
        for i in range(len(self.clips)):
            self.clipLengths[i]=self.clips[i].shape[0]

        #Compute number of valid sequences per clip
        self.clipLengths=(self.clipLengths-sequenceLength+1)
        print("Dataset contains {} unique sequences of length {}".format(np.sum(self.clipLengths),sequenceLength))
        if optimizeForSpeed:
            print("Optimizing data for faster sampling of sequences (may consume a lot of memory)...")
            numSequences=np.sum(self.clipLengths)
            self.allSequences=np.zeros([numSequences,sequenceLength,nPoseVars])
            seqIdx=0
            for clip in self.clips:
                for startFrame in range(clip.shape[0]-sequenceLength+1):
                    self.allSequences[seqIdx]=clip[startFrame:startFrame+sequenceLength]
                    seqIdx+=1
            assert(self.allSequences.shape[0]==seqIdx)
            self.allSequences=normalizeSequences(self.allSequences,self.normalizationPoseTime)
            if localPoseCoordinates:
                self.allSequences=globalToLocal(self.allSequences)
            print("Optimized dataset size:",self.allSequences.size)
        else:
            #Sum-normalize to get a discrete PDF for sampling clips and then sequences
            clipProbs=self.clipLengths/np.sum(self.clipLengths)
            #construct a discrete PDF object for sampling
            self.distrib = rv_discrete(values=(range(clipProbs.shape[0]), clipProbs))


    #Returns a batch of sequences usable for DRMM training
    def sampleSequences(self,numSequences):
        if self.optimizeForSpeed:
            indices=np.random.randint(0,self.allSequences.shape[0],size=numSequences)
            return self.allSequences[indices]

        else:
            clips=self.distrib.rvs(size=numSequences)
            result=np.zeros([numSequences,self.sequenceLength,self.nVarsPerFrame])
            for i in range(numSequences):
                clipIdx=clips[i]
                clipStart=np.random.randint(0,self.clipLengths[clipIdx])
                result[i]=self.clips[clipIdx][clipStart:clipStart+self.sequenceLength]
            return normalizeSequences(result,self.normalizationPoseTime)

def claWithoutLimitReset(ax):
    xlim = ax.get_xlim3d()
    ylim=ax.get_ylim3d()
    zlim=ax.get_zlim3d()
    ax.clear()
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)

if __name__ == '__main__':
    # Parse command line arguments
    #argv = sys.argv
    #args = parse_args(argv)
    #if args.debug:
    #    print(args)
    #main(args)
    #Best clips:
    #visualize_csv("mocapdata/walk1_subject5.bvh.csv",frameSkip=2)  #many directions, not too slow, determined
    #visualize_csv("mocapdata/walk3_subject5.bvh.csv",frameSkip=4)  #half crouched walks in all directions
    #visualize_csv("mocapdata/run1_subject5.bvh.csv",frameSkip=4)  #many directions, sideways too

    #visualize_csv("mocapdata/sprint1_subject4.bvh.csv",frameSkip=2) #fastest runs
    #visualize_csv("mocapdata/run2_subject4.bvh.csv",frameSkip=2)    #good medium speed runs, not too bouncy or "jogging in place"

    #Walk 2 clips all have: sassy, strutting, old man crouch, lady, drunk, hurt, staggering (push recovery?), on knees, energetic hops, soldier
    #They mostly differ slightly regarding style and how low the character crouches
    #visualize_csv("mocapdata/walk2_subject1.bvh.csv",frameSkip=10)   #many posing directions
    #visualize_csv("mocapdata/walk2_subject3.bvh.csv",frameSkip=10)   #crouches all the way to knees
    #visualize_csv("mocapdata/walk2_subject4.bvh.csv",frameSkip=10)   #not much crouching at all

    #visualize_csv("mocapdata/walk3_subject1.bvh.csv",frameSkip=10) #somewhat neutral walks, also includes sitting down and getting up
    #visualize_csv("mocapdata/walk3_subject2.bvh.csv",frameSkip=10) #somewhat neutral walks, lots of backward walking and also some running
    #visualize_csv("mocapdata/walk3_subject3.bvh.csv",frameSkip=10) #injured/bound walks and crawling on all fours
    #visualize_csv("mocapdata/walk3_subject4.bvh.csv",frameSkip=10) #normal walks, strutting hands on hips, lying down on back
    #visualize_csv("mocapdata/walk4_subject1.bvh.csv",frameSkip=10) #crouched walks



    visualize_csv("mocapdata/sprint1_subject2.bvh.csv",frameSkip=2)
    #visualize_csv("mocapdata/run1_subject2.bvh.csv",frameSkip=2)
    #visualize_csv("mocapdata/walk1_subject2.bvh.csv",frameSkip=2)
    #visualize_csv("mocapdata/walk3_subject2.bvh.csv",frameSkip=2)
    #dataset=MocapDataset("mocapdata/01_01.csv",sequenceLength=60)
    dataset=MocapDataset("mocapdata/sprint1_subject2.bvh.csv",sequenceLength=60,localPoseCoordinates=True)
    sequences=dataset.sampleSequences(100)
    sequences=localToGlobal(sequences)
    fig=plt.figure(figsize=[8.0,4.5],tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    setupAxes(ax)
    for i in range(100):
        for frameIdx in range(0,sequences.shape[1],2):
            claWithoutLimitReset(ax)
            setupAxes(ax)
            drawSkeleton(ax,sequences[i,frameIdx],color='black',alpha=1.0)
            #plt.draw()
            plt.pause(0.001)
        #dataset.visualizeSequence(sequences[i])
