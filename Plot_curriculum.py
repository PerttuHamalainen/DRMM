'''

Trains a DRMM with 2D Swiss roll data, and visualizes the samples with different model depths.

We also demonstrate how to use a Gaussian prior, box constraints, and linear inequality constraints.

'''


import numpy as np
import matplotlib.pyplot as pp
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #disable Tensorflow GPU usage, these simple graphs run faster on CPU
import tensorflow as tf
from matplotlib.patches import Ellipse, Rectangle
import DRMM
from DRMM import dataStream,DataIn
from DRMM import DRMM as Model

#parse arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--windowless', default=True,action='store_true')
parser.add_argument('--nIter', type=int, default=50000)
args = parser.parse_args()
nIter=args.nIter


#Visualization params
legendFontSize=9

#Training parameters
dataDim=2           #This example uses simple 2D data
nBatch=256           #Training minibatch size
learningRate=0.005


#Model parameters
nComponentsPerLayer=16
nLayers=4
prunedPercentage=0 #The percentage of least probable samples to prune before plotting

#Create Swiss roll data
x=[]
maxAngle=4.0*np.pi
for angle in np.arange(0,maxAngle,0.001):
    #swiss roll
    p=angle/maxAngle
    if np.random.uniform(0,1)<p:
        x.append(np.reshape(0.5*angle*np.array([np.sin(angle),np.cos(angle)]),[1,2]))
data=np.concatenate(x)

#A helper function for extracting a random data batch
def getDataBatch(nBatch):   
    return data[np.random.randint(data.shape[0], size=nBatch),:]


#Prepare plotting:
#We have 5 plots: samples without curriculum and after each of the 3 stages, and also a convergence graph
#that shows data log-likelihood progress with and without curriculum
subplotSize = 2.5
pp.figure(1, figsize=[4.5 * subplotSize, subplotSize])

for useCurriculum in [False,True]:
    #Create a TensorFlow session
    tf.reset_default_graph()
    sess=tf.Session()

    #Define the input data type. Note: the defaults are useBoxConstraints=False, useGaussianPrior=False, and
    #maxInequalities=0, which saves some memory and compute.
    inputStream=dataStream("continuous",               #This example uses continuous-valued data.
                         shape=[None,dataDim])          #The yet unknown batch size in the first value

    #Create model. Note: the constructor takes in the TensorFlow Session, but the model interface is otherwise
    #designed to be framework-agnostic, to prepare for upcoming TF2 and PyTorch implementations
    DRMM.useCurriculum=useCurriculum
    model=Model(sess=sess,
                nLayers=nLayers,
                nComponentsPerLayer=nComponentsPerLayer,
                inputs=inputStream,
                initialLearningRate=learningRate)

    #Initialize
    tf.global_variables_initializer().run(session=sess)
    model.init(getDataBatch(256))  #Data-dependent initialization

    #Helper for visualizing samples using the current model
    def showSamplesAndData():
        #Hide ticks
        ax=pp.gca()
        pp.setp(ax.get_xticklabels(), visible=False)
        pp.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        #Helper to ensure all plots are similarly scaled
        def setPlotLimits():
            pp.xlim(-6.4,4.8)
            pp.ylim(-5.6,7.1)

        #Plot input data
        markerSize=2
        pp.scatter(data[:,0],data[:,1],color='b',label='Training data',marker='.',s=markerSize,zorder=-1)

        #Sample and plot. In unconditional sampling, we only need to specify the number of samples
        samples=model.sample(nSamples=1024)
        samples=samples[:samples.shape[0]*(100-prunedPercentage)//100]

        pp.scatter(samples[:,0],samples[:,1],color='black',label='Samples',marker='.')
        setPlotLimits()

    #Optimize
    logps=[]
    for i in range(nIter):
        '''
        The train method performs a single EM step. 
        The method takes in a batch of data and a training phase variable in range 0...1.
        The latter is used for sweeping the learning rate and E-step \rho, as described in the paper.
        '''
        info=model.train(i/nIter,getDataBatch(nBatch))

        #Print progress
        if i%100==0 or i==nIter-1:
            logp=np.mean(model.getLogP(inputs=DataIn(data=data,mask=np.ones_like(data)))) #evaluate log-likelihood of all data
            print("\rIteration {}/{}, phase {:.3f} Loss {:.3f}, logp {:.3f} learning rate {:.6f}, precision {:.3f}".format(i,nIter,i/nIter,info["loss"],logp,info["lr"],info["rho"]),end="")
            logps.append(logp)

        if useCurriculum:
            if i==nIter//3:
                pp.subplot(1,5,2)
                showSamplesAndData()
                pp.title("Curriculum stage 1")
                if not args.windowless:
                    pp.pause(0.001)
            elif i==nIter//3*2:
                pp.subplot(1,5,3)
                showSamplesAndData()
                pp.title("Curriculum stage 2")
                if not args.windowless:
                    pp.pause(0.001)

    #Visualize final results
    if useCurriculum:
        pp.subplot(1,5,4)
        showSamplesAndData()
        pp.title("Curriculum stage 3")
        if not args.windowless:
            pp.pause(0.001)
    else:
        pp.subplot(1,5,1)
        showSamplesAndData()
        pp.title("No curriculum")
        pp.legend(fontsize=legendFontSize,framealpha=0.95)
        if not args.windowless:
            pp.pause(0.001)

    #Visualize convergence
    pp.subplot(1,5,5)
    if useCurriculum:
        stageLen=len(logps)//3
        stageIter=stageLen*100
        pp.plot(np.linspace(1, stageIter, stageLen), logps[:stageLen], label="Stage 1")
        pp.plot(np.linspace(stageIter+1, 2*stageIter, stageLen), logps[stageLen:stageLen*2], label="Stage 2")
        pp.plot(np.linspace(2*stageIter+1, nIter, stageLen), logps[stageLen*2:stageLen*3], label="Stage 3")
        #pp.plot(np.linspace(1, nIter, len(logps)), logps, label="With curriculum" if useCurriculum else "No curriculum")
    else:
        pp.plot(np.linspace(1,nIter,len(logps)),logps,label="No curriculum")
    pp.xlabel("Adam steps")
    pp.title("Log-likelihood")
    #pp.ylabel("Log-likelihood")
    #pp.title("Convergence")
    pp.legend(fontsize=legendFontSize)

    #Display plot without waiting
    if not args.windowless:
        pp.pause(0.001)

#Display plot, waiting for the user to close it
pp.tight_layout()
pp.savefig("images/curriculum.png",dpi=200)
if os.path.exists("PNGCrop.py"):
    import PNGCrop
    PNGCrop.crop("images/curriculum.png")
if not args.windowless:
    pp.show()

