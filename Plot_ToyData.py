import numpy as np
import random
import os
import matplotlib.pyplot as pp
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #disable Tensorflow GPU usage, these simple graphs run faster on CPU
import tensorflow as tf
import DRMM as DRMM
from matplotlib import rc

#use LaTeX fonts. Comment out if problems.
#rc('text', usetex=True)
#rc('font',**{'family':'serif','serif':['Times']})

#parse arguments
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--windowless', default=True,action='store_true')
parser.add_argument('--nIter', type=int, default=50000)
args = parser.parse_args()
nIter=args.nIter

#plot parameters
twoTitleRows=True
nDensityPlotSamples=32
fontsize=16 if twoTitleRows else 10
legendFontSize=11
plotCellHeight=2.5
plotCellWidth=2.2
pp.rcParams.update({'font.size': fontsize})
markerSize=4
nDatasets=2
datasetLayerAmounts=np.array([3,5])
datasetComponents=[16,3]
nPlotCols=np.sum(datasetLayerAmounts)
plotLimits=[1.1,1.1]
resPlotLimits=[1.1,1.1]
densityReso=512
plotConditionalSamples=True
DRMM.addLastLayerResidualNoise=True #we add the last layer residual noise to make the sampling consistent with the density plots
nBatch = 64
fastMode=True

#Loop over datasets, train a model, plot samples and densities
for datasetIdx in range(nDatasets):
    #helper
    def threshold(x,val):
        return tf.clip_by_value(tf.sign(x-val),0.0,1.0)


    #Create toy data
    if datasetIdx==1:
        x=[]

        def sierpinski(x0,x1,x2,data,depth=7):
            if depth==0:
                data.append(x0)
                data.append(x1)
                data.append(x2)
            else:
                depth-=1
                sierpinski(x0,0.5*(x0+x1),0.5*(x0+x2),data,depth)
                sierpinski(x1,0.5*(x1+x0),0.5*(x1+x2),data,depth)
                sierpinski(x2,0.5*(x2+x0),0.5*(x2+x1),data,depth)

        def pointOnUnitCircle(angle):
            return np.array([np.sin(angle),np.cos(angle)])
        sierpinski(pointOnUnitCircle(0),pointOnUnitCircle(1.0/3.0*2.0*np.pi),pointOnUnitCircle(2.0/3.0*2.0*np.pi),x)
        data=np.array(x)
        middle=0.5*(np.max(data,axis=0,keepdims=True)+np.min(data,axis=0,keepdims=True))
        data-=middle
        #span=np.max(data,axis=0,keepdims=True)-np.min(data,axis=0,keepdims=True)
        span=np.max(data)-np.min(data)
        data/=0.5*span
        dataDim=2
    else:
        ## set parameters
        #length_phi = 4.0*np.pi   #length of swiss roll in angular direction
        #sigma = 0.0       #noise strength
        #m = 10000         #number of samples

        ## create dataset
        #phi = length_phi*np.random.rand(m)
        #xi = np.random.rand(m)
        #X = 1./6*(phi + sigma*xi)*np.sin(phi)
        #Y = 1./6*(phi + sigma*xi)*np.cos(phi)

        #data = np.array([X, Y]).transpose()
        x=[]
        maxAngle=4.0*np.pi
        for angle in np.arange(0,maxAngle,0.001):
            #swiss roll
            p=angle/maxAngle
            if np.random.uniform(0,1)<p:
                x.append(np.reshape(0.5*angle*np.array([np.sin(angle),np.cos(angle)]),[1,2]))
        data=np.concatenate(x)

        span=np.max(data,keepdims=True)-np.min(data,keepdims=True)
        middle=0.5*(np.max(data,axis=0,keepdims=True)+np.min(data,axis=0,keepdims=True))
        data-=middle
        data/=0.5*span
        dataDim=2

    #init plotting and params
    nLayers=datasetLayerAmounts[datasetIdx]
    nComponentsPerLayer=datasetComponents[datasetIdx]
    plotBaseIdx=0 if datasetIdx==0 else datasetLayerAmounts[0]
    pp.figure(1,figsize=[nPlotCols*plotCellWidth,2.0*plotCellHeight])
    densityRange = np.linspace(-plotLimits[datasetIdx], plotLimits[datasetIdx], densityReso)
    densityCoords=np.zeros([densityReso*densityReso,2])
    idx=0
    for dx in densityRange:
        for dy in densityRange:
            densityCoords[idx]=[dy,-dx]
            idx+=1


    #Visualize
    for layerIdx in range(nLayers):
        if layerIdx==0 or (not fastMode):
            # create model
            tf.reset_default_graph()
            sess = tf.Session()
            # tf.set_random_seed(0)
            inputStream = DRMM.dataStream("continuous", shape=[None, dataDim], maxInequalities=1)
            model = DRMM.DRMM(sess, nLayers=nLayers if fastMode else layerIdx+1, nComponentsPerLayer=nComponentsPerLayer, inputs=inputStream, initialLearningRate=0.005)

            # Initialize
            tf.global_variables_initializer().run(session=sess)
            model.init(data)  # data dependent initialization: initializes Gaussian component means and the \sigma for each layer based on a data batch

            # Optimize
            for i in range(nIter + 1):
                batchData = data[np.random.randint(data.shape[0], size=nBatch), :]
                batchMask = np.ones([nBatch, dataDim])
                info = model.train(i / nIter, batchData)
                if i % 100 == 0:
                    logp = np.mean(model.getLogP(
                        inputs=DRMM.DataIn(data=data, mask=np.ones_like(data))))  # evaluate log-likelihood of all data
                    print("Iteration {}, Loss {:.3f}, Logp {:.3f}, learning rate {}".format(i, info["loss"], logp, info["lr"]))


        def hideTicks():
            ax=pp.gca()
            pp.setp(ax.get_xticklabels(), visible=False)
            pp.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

        ###FIRST PLOT: Training data and samples
        pp.figure(1)
        pp.subplot(2,nPlotCols,plotBaseIdx+layerIdx+1)
        pp.cla()

        if twoTitleRows:
            if layerIdx+1==1:
                pp.title("{} components\n{} layer".format(nComponentsPerLayer,layerIdx+1),fontsize=fontsize)
            else:
                pp.title("{} components\n{} layers".format(nComponentsPerLayer,layerIdx+1),fontsize=fontsize)
        else:
            pp.title("{} components, depth {}".format(nComponentsPerLayer,layerIdx+1),fontsize=fontsize)

        #Plot input data
        pp.scatter(data[:,0],data[:,1],color='blue',label='Training data',marker='.',s=markerSize*0.5)

        #Sample with inequality constraint: Constraint ieqWeight=1, known variables mask all zeros
        samples=model.sample(nSamples=1000,inputs=DRMM.DataIn(ieqs=[{"a":np.array([1.0,0.2]),"b":0.0}]),sampledLayer=layerIdx)
        pp.scatter(samples[:,0],samples[:,1],color='black',label='Samples, $ax+by+c>0$',marker='.',s=markerSize)

        #Sample y conditional on x
        if plotConditionalSamples:
            nCond=100
            xTarget=0.5                                  #target for x
            xIndex=0                                    #x values in column 0
            inputData=np.zeros([nCond,dataDim])         #initialize input data batch to zero
            mask=np.zeros([nCond,dataDim])              #initialize masks to zero (all variables unknown)
            inputData[:,xIndex]=xTarget                  #set the known variable's value
            mask[:,xIndex]=1                            #set the known variable's mask to 1

            #Generate and plot conditional samples.
            #In this case, we need to feed in the input data and known variables mask.
            #nSamples is not needed, because it is specified by the data shape.
            samples=model.sample(inputs=DRMM.DataIn(data=inputData,mask=mask),sampledLayer=layerIdx)
            pp.scatter(samples[:,0],samples[:,1],color='red',label='Samples, $x+c=0$',marker='.',s=markerSize)


        #if layerIdx==1 and datasetIdx==0:
        #    pp.legend(fontsize=legendFontSize,framealpha=0.95)
        hideTicks()
        pp.xlim(-plotLimits[datasetIdx],plotLimits[datasetIdx])
        pp.ylim(-plotLimits[datasetIdx],plotLimits[datasetIdx])

        ###SECOND PLOT: probability density
        pp.subplot(2,nPlotCols,nPlotCols+plotBaseIdx+layerIdx+1)
        p=0.0
        for _ in range(nDensityPlotSamples): #average over 100 samples (the density evaluations are stochastic due to the sampled latents)
            p+=np.exp(model.getLogP(inputs=DRMM.DataIn(data=densityCoords,mask=np.ones_like(densityCoords)),layerIdx=layerIdx))
        p=np.reshape(p,[densityReso,densityReso])

        #p=np.power(p,0.7) #gamma correction for visualization
        pp.imshow(p)
        hideTicks()


        pp.tight_layout(pad=0)
        if not args.windowless:
            pp.draw()
            pp.pause(0.001)

pp.figure(1)
pp.savefig("images/toydata.png",dpi=200) 
#pp.savefig("images/toydata.pdf")
if not args.windowless:
    pp.show()
