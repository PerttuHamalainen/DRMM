'''

This tutorial trains a hierarchical DRMM with MNIST image data, and visualizes both conditional and unconditional samples.

'''


import numpy as np
import matplotlib.pyplot as pp
import random
import os
import tensorflow as tf
from DRMM import DRMMBlockHierarchy,dataStream,DataIn
from tensorflow.examples.tutorials.mnist import input_data
import time


#Plotting parameters
imageGridSize=8
subplotSize=3.5   

#Training parameters
nIter=50000                 #Number of training iterations (minibatches). Set this to a higher value, e.g, 200000 for higher quality results.
nBatch=imageGridSize**2     #Training minibatch size
dataset="MNIST"             #Tensorflow dataset, e.g., "MNIST" or "CIFAR10"
modelFileName="trainedmodels/tutorial_{}".format(dataset)
train=not os.path.isfile(modelFileName+".index")   #by default, we do not train again if saved model found. To force retraining, set train=True

#Inference parameters
nSampleBatch=256

#Load MNIST data
tfData = input_data.read_data_sets("{}_data/".format(dataset), one_hot=True)
if dataset=="MNIST":
    RESO=28
    CHANNELS=1
elif dataset=="CIFAR10":
    RESO=32
    CHANNELS=3
else:
    raise Exception("Unsupported dataset")

#Helper for querying a batch of training images, reshaped to [nBatch,width,height,channels]
def getDataBatch(nBatch):
    data, _ = tfData.train.next_batch(nBatch)
    return np.reshape(data,[-1,RESO,RESO,CHANNELS])

#Helper for querying a batch of test images, reshaped to [nBatch,width,height,channels]
def getTestDataBatch(nBatch):
    data, _ = tfData.test.next_batch(nBatch)
    return np.reshape(data,[-1,RESO,RESO,CHANNELS])


#Helper for showing a batch of images as an array (assuming batch shape [nBatch,width,height,nChannels], with nChannels either 1 or 3)
def showImages(images,nCols=None,nRows=None):
    if (nCols is None) or (nRows is None):
        nCols=int(np.round(np.sqrt(images.shape[0])))
        nRows=nCols
    w=images.shape[1]
    h=images.shape[2]
    imageArr=np.zeros([w*nCols,h*nRows,3])
    for x in range(nCols):
        for y in range(nRows):
            i=x+y*nCols
            if i<images.shape[0]:
                imageArr[x*w:(x+1)*w,y*h:(y+1)*h]=images[i,:,:]
    pp.imshow(imageArr)
    ax=pp.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


#PLOT 1: show training data
pp.figure(1,figsize=[subplotSize*3.0,subplotSize],tight_layout=True)
pp.subplot(1,3,1)
pp.cla()
showImages(getDataBatch(nBatch))
pp.title("Training data")


#Init tf
tf.reset_default_graph()
sess=tf.Session()
tf.set_random_seed(int(time.time()))


'''
The hierarchical model has DRMM blocks that model 3x3 pixel patches, while decreasing the modeled image resolution through striding
As the modeled data grows more complex on each level of the hierarchy, we gradually increase the DRMM layer counts.
The last DRMM block models the joint distribution of the latents produced by the segment models.
'''
model=DRMMBlockHierarchy(sess,
                         inputs=dataStream("continuous",shape=[None,RESO,RESO,CHANNELS]),
                         blockDefs=[
                         {"nClasses":128,"nLayers":1,"kernelSize":[3,3],"stride":[2,2]},   #in 28x28, out 14x14
                         {"nClasses":128,"nLayers":2,"kernelSize":[3,3],"stride":[2,2]},   #in 14x14, out 7x7
                         ],
                         lastBlockClasses=128,
                         lastBlockLayers=4,
                         train=train,
                         initialLearningRate=0.005)
print("Total model parameters: ",model.nParameters)

#Train or load model
saver = tf.train.Saver()
if not train:
    saver.restore(sess,modelFileName)
else:
    #Initialize 
    print("Initializing...")
    tf.global_variables_initializer().run(session=sess)
    model.init(getDataBatch(nBatch)) #Data-driven init with a random batch

    #Train
    print("Training...")
    for i in range(nIter):
        info=model.train(i/nIter,getDataBatch(nBatch))
        if i%10==0:
            print("Stage {}/{}, Iteration {}/{}, Loss {:.3f}, learning rate {:.6f}, precision {:.3f}".format(
                info["stage"],info["nStages"],
                i,nIter,
                info["loss"],
                info["lr"],
                info["rho"]),end="\r")
        #Uncomment if you want to visualize samples while training
        #if i%1000==0:
        #    samples=model.sample(nSampleBatch) 

        #    #Ensure that modeling inaccuracy does not lead to invalid pixel values
        #    samples=np.clip(samples,0,1)   

        #    #Plot
        #    pp.subplot(1,3,2)
        #    pp.cla()
        #    showImages(samples[:nBatch])
        #    pp.title("Unconditional samples")
        #    pp.pause(0.001)
    if not os.path.exists('trainedmodels'):
        os.makedirs('trainedmodels')
    saver.save(sess,modelFileName)

#PLOT 2: Visualize unconditional samples
#Generate samples
samples=model.sample(nSampleBatch) 

#Ensure that modeling inaccuracy does not lead to invalid pixel values
samples=np.clip(samples,0,1)   

#Plot
pp.subplot(1,3,2)
pp.cla()
showImages(samples[:nBatch])
pp.title("Unconditional samples")

#PLOT 3: Visualize samples conditioned on known pixels
testImages=getTestDataBatch(nBatch)

#Visualize the unknown pixels with intensity 0.5
testImages[:,RESO//2:]=0.5 

#Initialize samples to zero
samples=np.zeros_like(testImages)

#Loop over test images (one row of the image grid) and generate samples
for i in range(imageGridSize):
    #Image sampling uses backward sampling, which requires the same conditioning info for all samples
    #Thus, we make one sampling input batch of each test image
    testData=np.repeat(testImages[i:i+1],axis=0,repeats=nSampleBatch)

    #Set the mask to zero for the unknown pixels
    testMask=np.ones_like(testData)
    testMask[:,RESO//2:]=0.0

    #Sample
    tempSamples=model.sample(inputs=DataIn(data=testData,mask=testMask)) 
    tempSamples=np.clip(tempSamples,0,1)   #ensure that modeling inaccuracy does not lead to invalid pixels

    #Copy samples to the visualized array
    samples[i]=testImages[i]
    for j in range(imageGridSize-1):
        samples[(j+1)*imageGridSize+i]=tempSamples[j]

#Plot
pp.subplot(1,3,3)
pp.cla()
showImages(samples)
pp.title("Conditional samples")
pp.savefig("images/tutorial_images.png")
pp.show()