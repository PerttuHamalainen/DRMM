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
from DRMM import dataStream,DRMM,DataIn


#Training parameters
dataDim=2           #This example uses simple 2D data
nIter=40000         #Number of training iterations (minibatch EM steps)
nBatch=64           #Training minibatch size

#Model parameters
bwdSampling=False   #Set to true to test backward sampling instead of forward sampling
nComponentsPerLayer=16
nLayers=4

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

#Create a TensorFlow session
sess=tf.Session()

#Define the input data type. Note: the defaults are useBoxConstraints=False, useGaussianPrior=False, and
#maxInequalities=0, which saves some memory and compute.
inputStream=dataStream("continuous",               #This example uses continuous-valued data.
                         shape=[None,dataDim],          #The yet unknown batch size in the first value
                         useBoxConstraints=True,        #We use box constraints
                         useGaussianPrior=True,         #We use a Gaussian prior
                         maxInequalities=2)             #We use up to 2 inequality constraints

#Create model. Note: the constructor takes in the TensorFlow Session, but the model interface is otherwise
#designed to be framework-agnostic, to prepare for upcoming TF2 and PyTorch implementations 
model=DRMM(sess=sess,
                nLayers=nLayers,
                nComponentsPerLayer=nComponentsPerLayer,
                inputs=inputStream,
                initialLearningRate=0.005,
                useBwdSampling=bwdSampling)

#Initialize
tf.global_variables_initializer().run(session=sess)
model.init(getDataBatch(256))  #Data-dependent initialization 

#Optimize
for i in range(nIter):
    '''
    The train method performs a single EM step. 
    The method takes in a batch of data and a training phase variable in range 0...1.
    The latter is used for sweeping the learning rate and E-step \rho, as described in the paper.
    '''
    info=model.train(i/nIter,getDataBatch(nBatch))

    #Print progress
    if i%100==0 or i==nIter-1:
        print("Iteration {}/{}, phase {:.3f} Loss {:.3f}, learning rate {:.6f}, precision {:.3f}".format(i,nIter,i/nIter,info["loss"],info["lr"],info["rho"]),end="\r")

    #Visualize progress
    if i%1000==0 or i==nIter-1:
        #Helper to ensure all plots are similarly scaled
        def setPlotLimits():
            pp.xlim(-6.4,4.8)
            pp.ylim(-5.6,7.1)

        #Plot input data
        subplotSize=2.5
        pp.figure(1,figsize=[5.0*subplotSize,subplotSize],tight_layout=True)
        pp.clf()
        pp.subplot(1,5,1)   
        markerSize=2
        pp.scatter(data[:,0],data[:,1],color='b',label='input',marker='.',s=markerSize,zorder=-1)

        #Sample and plot. In unconditional sampling, we only need to specify the number of samples
        samples=model.sample(nSamples=1024)

        pp.scatter(samples[:,0],samples[:,1],color='black',label='samples, p(x,y)',marker='.')
        pp.title("Unconditional samples")
        setPlotLimits()

        #Prepare sampling conditional on a desired x coordinate
        nCond=256                                   #number of conditional samples
        target=2.5                                  #target for x
        xIndex=0                                    #x values in column 0
        inputData=np.zeros([nCond,dataDim])         #initialize input data batch to zero
        mask=np.zeros([nCond,dataDim])              #initialize masks to zero (all variables unknown)
        inputData[:,xIndex]=target                  #set the known variable's value
        mask[:,xIndex]=1                            #set the known variable's mask to 1 

        #Generate and plot conditional samples. 
        #In this case, we need to feed in the input data and known variables mask.
        #nSamples is not needed, because it is specified by the data shape.
        samples=model.sample(inputs=DataIn(data=inputData,mask=mask))

        pp.subplot(1,5,2)
        pp.scatter(data[:,0],data[:,1],color='b',marker='.',s=markerSize,zorder=-1)
        pp.plot([target,target],[-10,10],color="gray",lw=3.0,zorder=-0.5) #visualize the target x 
        pp.scatter(samples[:,0],samples[:,1],color='black',marker='.')
        setPlotLimits()
        pp.title("Samples, p(y | x={})".format(target))
        setPlotLimits()

        #Generate and plot samples with inequality constraints of type dot(a,x)+b>0.
        #The inequalities are defined as a list of dictionaries. 
        #The number of dictionaries must be less than equal to the maxInequalities parameter passed to the model constructor.
        ieqs=[{"a":np.array([1.0,0.2]),"b":2.0},{"a":np.array([-1.0,-0.2]),"b":2.0}]
        samples=model.sample(nSamples=nCond,inputs=DataIn(ieqs=ieqs))

        pp.subplot(1,5,3)
        pp.scatter(data[:,0],data[:,1],color='b',marker='.',s=markerSize,zorder=-1)
        for ieq in ieqs:
            #Plot the constraint boundary line
            #a0*x0+a1*x1+b=0 => x1=(-a0*x0-b)/a1
            x0=np.array([-5.0,5.0])
            a0=ieq["a"][0]
            a1=ieq["a"][1]
            x1=(-a0*x0-ieq["b"])/a1
            pp.plot(x0,x1,color="gray",lw=3.0,scalex=False,scaley=False,zorder=-0.5) 
        pp.scatter(samples[:,0],samples[:,1],color='black',marker='.')
        pp.title("Inequality constraints")
        setPlotLimits()

        #Generate and plot samples with box constraints, i.e., limits for maximum and minimum values
        minValues=np.array([-5.0,-4.0])
        maxValues=np.array([3.0,4.0])
        samples=model.sample(nSamples=nCond,inputs=DataIn(minValues=minValues,maxValues=maxValues))

        pp.subplot(1,5,4)
        pp.scatter(data[:,0],data[:,1],color='b',marker='.',s=markerSize,zorder=-1)
        pp.gca().add_patch(Rectangle(minValues,width=(maxValues-minValues)[0],height=(maxValues-minValues)[1],facecolor='none',edgecolor="gray",lw=3.0,zorder=-0.5)) 
        pp.scatter(samples[:,0],samples[:,1],color='black',marker='.')
        pp.title("Box constraints")
        setPlotLimits()

        #Generate and plot samples with a Gaussian prior        
        priorMean=np.array([0.0,0.0])
        priorSd=np.array([3.0,1.0])
        samples=model.sample(nSamples=nCond,inputs=DataIn(priorMean=priorMean,priorSd=priorSd))

        pp.subplot(1,5,5)
        pp.scatter(data[:,0],data[:,1],color='b',marker='.',s=markerSize,zorder=-1)
        pp.gca().add_patch(Ellipse(priorMean,width=4.0*priorSd[0],height=4.0*priorSd[1],facecolor='none',edgecolor="gray",lw=3.0,zorder=-0.5)) #4.0 multiplier to make the ellipse correspond to two standard deviations
        pp.scatter(samples[:,0],samples[:,1],color='black',marker='.')
        pp.title("Gaussian prior")
        setPlotLimits()

        #Display plot without waiting
        pp.pause(0.001)

#Display plot, waiting for the user to close it
pp.savefig("images/tutorial_swissroll.png",dpi=200)
pp.show()

