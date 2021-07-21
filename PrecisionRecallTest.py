
import numpy as np
import random
import os
import matplotlib.pyplot as pp
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" #disable Tensorflow GPU usage, these simple graphs run faster on CPU
import tensorflow as tf
import DRMM as DRMM
from skimage.util import view_as_blocks
from precision_recall import knn_precision_recall_features
import MocapUtils as mocap
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datasetIdx', type=int, default=1)
parser.add_argument('--modelIdx', type=int, default=0)
parser.add_argument('--nIter', type=int, default=50000)
parser.add_argument('--nEval', type=int, default=20000)

args = parser.parse_args()
datasetIdx=args.datasetIdx
modelIdx=args.modelIdx
nIter=args.nIter

nBatch=64
initialLearningRate=0.002
#datasets=["IK (arm)"] 
datasets=["IK (fullbody)","Motion Capture"]
nTargetEvalSamples=args.nEval

#Returns squared distance matrix D with elements d_ij = | a_i - b_j|^2, where a_i = A[i,:] and b_j=B[j,:]
def pairwiseSqDistances(A,B):
    #d_ij=(a_i-b_j)'(a_i-b_j) = a_i'a_i - 2 a_i'b_j + b_j'b_j
    #D = [a_0'a_0, a_1'a_1, ...] - 2 AB' + [b_0'b_0, b_1'b_1, ...]',  assuming broadcasting
    #D = A_d - 2 AB' + B_d
    A_d=np.sum(A * A,axis=1,keepdims=True)
    B_d=np.reshape(np.sum(B * B,axis=1),[1,B.shape[0]])
    return np.clip(A_d - 2 * np.matmul(A,np.transpose(B)) + B_d,0,np.inf)  #relu to ensure no negative results due to computational inaccuracy

def modifiedHausdorffDistance(A,B):
    sqDist=pairwiseSqDistances(A,B)
    return np.sqrt(np.sum(np.min(sqDist,axis=0))+np.sum(np.min(sqDist,axis=1)))

def numDrmmParameters(dataDim,nLayers,nComponentsPerLayer):
    nParameters=0
    layerInputVars=dataDim
    for layerIdx in range(nLayers):
        nParameters+=1 #scalar variance parameter
        nParameters+=layerInputVars*nComponentsPerLayer  #Gaussian means or class prototypes
        nParameters+=nComponentsPerLayer #marginal probabilities
        layerInputVars+=nComponentsPerLayer
    return nParameters


plotIdx=0
dataset=datasets[datasetIdx]
#Load or create data
if dataset=="Swissroll 3D":
    print("Creating 3D swissroll data")
    x=[]
    noiseSd=0.0
    for angle in np.arange(0,4.0*np.pi,0.001):
        #swiss roll
        x.append(np.reshape(0.5*angle*np.array([np.sin(angle),np.cos(angle)])+np.random.normal(0,noiseSd,size=[2]),[1,2]))
        #circle
        #x.append(np.reshape(np.array([np.sin(angle),np.cos(angle)])+np.random.normal(0,noiseSd,size=[2]),[1,2]))
        #sine wave
        #x.append(np.reshape(np.array([angle,np.cos(angle)])+np.random.normal(0,noiseSd,size=[2]),[1,2]))
    data=np.concatenate(x)
    data=np.concatenate([data,np.random.uniform(-2,2,size=[data.shape[0],1])],axis=1)
elif dataset=="Sierpinski 2D":
    x=[]

    def sierpinski(x0,x1,x2,data,depth=8):
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
elif dataset=="IK (arm)":
    print("Loading data")
    dataFile="./IKTest/arm_data.npy"
    data=np.load(dataFile)
elif dataset=="IK (fullbody)":
    print("Loading data")
    dataFile="./IKTest/fullbody_data.npy"
    data=np.load(dataFile)
elif dataset == "Motion Capture":
    print("Loading Motion Capture Data")
    mocapData = mocap.MocapDataset("mocapdata/laforge_locomotion_nosliding.zip",
                                 sequenceLength=30,
                                 optimizeForSpeed=True)
    data=mocapData.allSequences.reshape([mocapData.allSequences.shape[0],-1])
else:
    raise Exception("Invalid dataset")
dataDim=data.shape[1]
print("Dataset has {} vectors of {} variables".format(data.shape[0],data.shape[1]))
#if data.shape[0]>maxData:
#    data=data[:maxData]
#A helper function for extracting a random data batch
def getDataBatch(nBatch):
    return data[np.random.randint(data.shape[0], size=nBatch),:]

layerAmounts=[1,2,3,4]



nLayers=layerAmounts[modelIdx]


#We test GMM with 64,128... components, and DRMM:s with the same (approximately) number of parameters
for modelSize in [64,128,256,512,1024]:
    if nLayers==1:
        nComponentsPerLayer=modelSize
    else:
        targetNumParams=numDrmmParameters(dataDim,1,modelSize)
        nComponentsPerLayer=4
        while (numDrmmParameters(dataDim,nLayers,nComponentsPerLayer)<targetNumParams):
            nComponentsPerLayer+=1
    nParameters=numDrmmParameters(dataDim,nLayers,nComponentsPerLayer)

    #Init tf
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    #create model
    if nParameters<2000: # or (datasetIdx==2 and nLayers==1):  #run small models on CPU (faster), and also run the datasetIdx==2 and nLayers==1 cases on CPU because they need a lot of memory and the dev. laptop GPU is only 6GB
        with tf.device('/cpu:0'):  
            model=DRMM.DRMM(sess=sess,
                            nLayers=nLayers,
                            nComponentsPerLayer=nComponentsPerLayer,
                            initialLearningRate=initialLearningRate,
                            inputs=DRMM.dataStream("continuous",shape=[None,dataDim]))
    else:
        model = DRMM.DRMM(sess=sess,
                          nLayers=nLayers,
                          nComponentsPerLayer=nComponentsPerLayer,
                          initialLearningRate=initialLearningRate,
                          inputs=DRMM.dataStream("continuous", shape=[None, dataDim]))
    assert(nParameters==model.nParameters)  #check that our parameter amount estimation was correct



    #Initialize
    tf.global_variables_initializer().run(session=sess)
    model.init(data[:min([2000,data.shape[0]])])

    #Optimize
    for i in range(nIter):
        info = model.train(i / nIter, getDataBatch(nBatch))
        # Print progress
        if i % 100 == 0 or i == nIter - 1:
            logp = np.mean(
                model.getLogP(inputs=DRMM.DataIn(data=getDataBatch(1024),mask=np.ones([1024,dataDim]))))  # evaluate log-likelihood of a large data batch
            print(
                "\rIteration {}/{}, phase {:.3f} Loss {:.3f}, logp {:.3f} learning rate {:.6f}, precision {:.3f}".format(
                    i, nIter, i / nIter, info["loss"], logp, info["lr"], info["rho"]), end="")

    #Evaluate
    nEvalSamples=min([data.shape[0],nTargetEvalSamples])
    print("\nGenerating {} samples".format(nEvalSamples))
    sampled_fetch=np.zeros([nEvalSamples,dataDim])
    nSampled=0
    while (nSampled<nEvalSamples):
        batchSize=min([10000,nEvalSamples-nSampled])
        sampled_fetch[nSampled:nSampled+batchSize]=model.sample(nSamples=batchSize)
        nSampled+=batchSize
    #print(sampled_fetch)
    print("Evaluating")
    if nEvalSamples<data.shape[0]:
        evalData=getDataBatch(nEvalSamples)
    else:
        evalData=data
    #evalData=data[:min([nEvalSamples, data.shape[0]])]
    logp = np.mean(model.getLogP(inputs=DRMM.DataIn(data=evalData,mask=np.ones_like(evalData))))
    with sess.as_default():
        #Precision and recall code from: https://github.com/kynkaat/improved-precision-and-recall-metric
        precrecall=knn_precision_recall_features(evalData,sampled_fetch,row_batch_size=10000)
    precision=precrecall['precision'][0]
    recall=precrecall['recall'][0]
    f1=2.0*(recall * precision) / (recall + precision + 1e-8)
    print("F1 {}, logp {}".format(f1,logp))
    logFileName="Results/benchmark_precrecall.csv"
    if not os.path.isfile(logFileName):
        logFile=open(logFileName,"w")
        logFile.write("dataset,datasetIdx,nLayers,nComponentsPerLayer,nParameters,precision,recall,f1,logp\n")
    else:
        logFile=open(logFileName,"a")
    #logFile.write("dataset,datasetIdx,nLayers,nComponentsPerLayer,sampleQuality")
    logFile.write("{},{},{},{},{},{},{},{},{}\n".format(dataset,datasetIdx,nLayers,nComponentsPerLayer,model.nParameters,precision,recall,f1,logp))
    logFile.close()
    #pp.close()



