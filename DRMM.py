import numpy as np
import random
import os
import copy
import tensorflow as tf
import itertools
from math import ceil

#Parameters that one may try to adjust
regularization=0.01            #\alpha in the paper
epsilon=1e-8                   #Laplace smoothing epsilon
fwdTruncation=0.05             #Forward pass membership sampling distribution truncation parameter
trainingFwdTruncation=0.5      #The same but during training. High truncation improves mode separation
sigmaType="scalar"             #The paper uses scalar sigma (component covariance) for simplicity, but the code now supports per-variable sigma, i.e., diagonal covariance

#Parameters that should not be touched. 
arrayMarginalMode="global"                 #"global": a single w_k shared by all patch DRMMs, "elementwise": the w_k not shared (instead, separate w_k for each extracted input patch location)
dataDependentInitializationMode="normal"   #Defines how components are initialized based on an initial batch of data. "normal" is the default, i.e., randomizing the means from a diagonal Gaussian approximation of the input
initialSdScale=0.1                         #\sigma of a layer's components is initially set to layer input stdev * initialSdScale
MStepPrecision=1.0                         #\rho for M-step. values other than 1 are only used for comparison in a paper image
addLastLayerResidualNoise=False            #A value True is only used for the paper's Figure 1.
modifiedEStep=True                         #A value of False is only used for comparison in a paper image.
reparameterizedSampling=False              #if True, we use reparameterized Gumbel-Softmax sampling of the latents. However, this seems to yield worse results
gumbelTemperature=0.1                      #Gumbel-Softmax temperature, only used if reparameterizedSampling==True
useCurriculum=True
bwdTruncation=0.0                          #\alpha_t in
sampleInBwdPass=False                      #If true, selects an experimental backward sampling procedure
UnetSkipStrength=1.0/3.0 if sampleInBwdPass else 0.0     #Fwd-bwd pass mixing coefficient, only used for the experimental backward sampling
trainingTemperature=1.0        #Temperature parameter that can be used similar to trainingFwdTruncation, but is not used by default


#Reparameterized Gumbel-Softmax helpers from Eric Jang's blog post: https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
#(We're using an old Tensorflow version that doesn't have these integrated)
def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def sampleReparameterized(logits, temperature=gumbelTemperature, hard=True):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y

#Helper: (partial) gradient stop.
def stopGradient(x,gradStopStrength):
    if isinstance(gradStopStrength,float) and gradStopStrength==1.0:
        return tf.stop_gradient(x)
    return gradStopStrength*tf.stop_gradient(x)+(1.0-gradStopStrength)*x

#Helper
def softmaxWithTemperature(x,temperature=None):
    if temperature is None:
        return tf.nn.softmax(x)
    if (isinstance(temperature,float) or isinstance(temperature,int)) and temperature==0:
        indices=tf.argmax(x,axis=-1)
        return tf.one_hot(indices=indices,depth=x.shape[-1].value)
    else:
        return tf.nn.softmax(x/temperature)

#Helper: Smoothing of a discrete pdf
def discretePdfSmooth(x,nCategories,c=epsilon):
    return (1.0-c)*x+c/nCategories

#Helper: Regularized (safe) log
def discretePdfLog(x,nCategories,c=epsilon):
    return tf.log(discretePdfSmooth(x,nCategories,c))

#Helper: Normalize a discrete pdf, or a tensor that packs multiple pdf:s in a single dimension (which happens when patches extracted of categorical image streams)
def discretePdfNormalize(pdf:tf.Tensor,nCategories,epsilon=epsilon):
    #First, handle the simple case where the last tensor dimension corresponds to the pdf
    shape=pdf.shape.as_list()
    if shape[-1]==nCategories:
        return pdf/(tf.reduce_sum(pdf,axis=-1,keepdims=True)+epsilon)
    #Handle the more complex case where the last tensor dimension packs multiple pdfs.
    #This happens, e.g., witch patches extracted from 2D or 1D arrays of categorical latents
    shape[0]=-1
    shape2=shape.copy()
    shape2[-1]=shape[-1]//nCategories
    shape2.append(nCategories)
    deinterleaved=tf.reshape(pdf,shape2)
    deinterleaved/=tf.reduce_sum(deinterleaved,axis=-1,keepdims=True)+epsilon
    return tf.reshape(deinterleaved,shape)

#Helper
def discretePdfBatchAverage(pdf,nCategories,epsilon=0,sampleWeights=None):
    if sampleWeights is None:
        averaged=tf.reduce_sum(pdf,axis=0,keepdims=True)
    else:
        averaged = tf.reduce_sum(sampleWeights*pdf, axis=0, keepdims=True)
    return discretePdfNormalize(averaged,nCategories=nCategories,epsilon=epsilon)

#Softmax that works for both a single tensor of logits and one that packs a patch of logits in image modeling
def streamSoftmax(logits:tf.Tensor,nCategories:int):
    #return tf.nn.softmax(logits)
    shape=logits.shape.as_list()
    if shape[-1]==nCategories:
        return tf.nn.softmax(logits)
    shape[0]=-1
    shape2=shape.copy()
    shape2[-1]=shape[-1]//nCategories
    shape2.append(nCategories)
    deinterleaved=tf.reshape(logits,shape2)
    deinterleaved=tf.nn.softmax(deinterleaved)
    return tf.reshape(deinterleaved,shape)

#Log-softmax that works for both a single tensor of logits and one that packs a patch of logits in image modeling
def streamLogSoftmax(logits:tf.Tensor,nCategories:int):
    #return tf.nn.softmax(logits)
    shape=logits.shape.as_list()
    if shape[-1]==nCategories:
        return tf.nn.log_softmax(logits)
    shape[0]=-1
    shape2=shape.copy()
    shape2[-1]=shape[-1]//nCategories
    shape2.append(nCategories)
    deinterleaved=tf.reshape(logits,shape2)
    deinterleaved=tf.nn.log_softmax(deinterleaved)
    return tf.reshape(deinterleaved,shape)

#Helper: "hard" softmax
def softmaxWithHardness(x,hardness=None):
    x=tf.nn.softmax(x,axis=-1)
    if hardness is None:
        return x
    indices=tf.argmax(x,axis=-1)
    x=(1.0-hardness)*x + hardness*tf.one_hot(indices=indices,depth=x.shape[-1].value)
    x/=tf.reduce_sum(x,axis=-1,keepdims=True)
    return x


#Linear inequality constraint
class IEQConstraint:
    def __init__(self,a,b,weight=1.0):
        self.a=a
        self.b=b
        self.weight=weight
    def copy(self):
        return IEQConstraint(self.a,self.b,self.weight)

#Box constraint
class BoxConstraint:
    def __init__(self,minValues,maxValues,minValueWeights,maxValueWeights):
        self.minValues=minValues
        self.maxValues=maxValues
        self.minValueWeights=minValueWeights
        self.maxValueWeights=maxValueWeights
    def copy(self):
        return BoxConstraint(self.minValues,self.maxValues,self.minValueWeights,self.maxValueWeights)

#Gaussian prior
class GaussianPrior:
    def __init__(self,mean,sd,weight):
        self.mean=mean
        self.sd=sd
        self.weight=weight
    def copy(self):
        return GaussianPrior(self.mean,self.sd,self.weight)

class DataIn:
    """A container for the data fed in as a DataStream, to simplify the user interface of DRMM models"""
    def __init__(self,data=None,mask=None,ieqs=None,minValues=None,maxValues=None,minValueWeights=None, maxValueWeights=None,priorMean=None,priorSd=None,priorWeight=None,weights=None):
        self.data=data
        self.mask=mask
        self.ieqs=ieqs
        self.minValues=minValues
        self.maxValues=maxValues
        self.minValueWeights=minValueWeights
        self.maxValueWeights=maxValueWeights
        self.priorMean=priorMean
        self.priorSd=priorSd
        self.priorWeight=priorWeight
        self.weights=weights
        
class DataStream:
    """
    Datastream class. Encapsulates the data vector x (self.tensor), known variables mask, and the The mask contains values in range 0...1, determining how much weight each sample gets in the class log probabilities
    """
    def __init__(self,tensor:tf.Tensor,type,mask:tf.Tensor=None,nCategories:int=0,ieqConstraints=None,gaussianPrior=None,boxConstraint=None,weights=None,logQ=None):
        self.tensor=tensor
        self.type=type
        assert(type=="continuous" or type=="discrete")
        assert(not(type=="discrete" and nCategories==0))
        self.mask=mask
        self.nCategories=nCategories
        self.ieqConstraints=ieqConstraints if ieqConstraints is not None else []
        self.boxConstraint=boxConstraint
        self.gaussianPrior=gaussianPrior
        self.weights=weights
        self.logQ=logQ
    def __repr__(self):
        return "DataStream: shape {}, type {}, mask {}, nCategories {}".format(self.tensor.shape,self.type,self.mask,self.nCategories)
    def __str__(self):
        return "DataStream: shape {}, type {}, mask {}, nCategories {}".format(self.tensor.shape,self.type,self.mask,self.nCategories)
    def copy(self):
        priorCopy=None if self.gaussianPrior is None else self.gaussianPrior.copy()
        bcCopy=None if self.boxConstraint is None else self.boxConstraint.copy()
        ieqCopy=[]
        for ieq in self.ieqConstraints:
            ieqCopy.append(ieq.copy())
        return DataStream(tensor=self.tensor,type=self.type,mask=self.mask,nCategories=self.nCategories,ieqConstraints=ieqCopy,gaussianPrior=priorCopy,boxConstraint=bcCopy,weights=self.weights,logQ=self.logQ)

def processStreamTensors(stream,processFunc,broadcastToBatch=False):
    # process data
    batchShape = tf.shape(stream.tensor)
    stream.tensor = processFunc(stream.tensor)
    # process known variables mask
    stream.mask = processFunc(stream.mask)
    # process logQ
    stream.logQ=processFunc(stream.logQ)
    # process weights
    stream.weights=processFunc(stream.weights)

    # process the Gaussian prior, broadcasting a single-sample prior to full batch to avoid math broadcasting problems later on
    def singleToBatch(tensor):
        if broadcastToBatch:
            return tf.broadcast_to(tensor, batchShape)
        else:
            return tensor

    if stream.gaussianPrior is not None:
        stream.gaussianPrior.mean = processFunc(singleToBatch(stream.gaussianPrior.mean))
        stream.gaussianPrior.sd = processFunc(singleToBatch(stream.gaussianPrior.sd))

        # process box constraints
    if stream.boxConstraint is not None:
        stream.boxConstraint.minValues = processFunc(singleToBatch(stream.boxConstraint.minValues))
        stream.boxConstraint.maxValues = processFunc(singleToBatch(stream.boxConstraint.maxValues))
        stream.boxConstraint.minValueWeights = processFunc(singleToBatch(stream.boxConstraint.minValueWeights))
        stream.boxConstraint.maxValueWeights = processFunc(singleToBatch(stream.boxConstraint.maxValueWeights))
    # process linear inequality constraints
    for c in stream.ieqConstraints:
        c.a = processFunc(singleToBatch(c.a))



def dataStream(dataType,shape,nCategories:int=0,useBoxConstraints=False,useGaussianPrior=False,maxInequalities=0):
    """Helper for creating a DataStream object and the Tensorflow placeholders needed for passing in the data, masks, constraints and priors."""
    assert(dataType=="continuous" or dataType=="discrete")
    inputTensor=tf.placeholder(dtype=tf.float32,shape=shape,name="in_data")
    maskTensor=tf.placeholder(dtype=tf.float32,shape=shape,name="in_mask")
    scalarShape=shape.copy()
    for i in range(len(scalarShape)):
        scalarShape[i]=1
    weightsTensor=tf.placeholder(dtype=tf.float32,shape=[None,1],name="in_weights")
    if maxInequalities>0:
        ieqs=[]
        for i in range(maxInequalities):
            a=tf.placeholder(dtype=tf.float32,shape=shape,name="in_ieq_a")
            b=tf.placeholder(dtype=tf.float32,shape=scalarShape,name="in_ieq_b")
            w=tf.placeholder(dtype=tf.float32,shape=[],name="in_ieq_w")
            ieq=IEQConstraint(a,b,w)
            ieqs.append(ieq)
    else:
        ieqs=None

    if useGaussianPrior:
        mean=tf.placeholder(dtype=tf.float32,shape=shape,name="in_prior_mean")
        sd=tf.placeholder(dtype=tf.float32,shape=shape,name="in_prior_sd")
        weight=tf.placeholder(dtype=tf.float32,shape=[],name="in_prior_weight")
        gaussianPrior=GaussianPrior(mean=mean,sd=sd,weight=weight)
    else:
        gaussianPrior=None
    if useBoxConstraints:
        minValues=tf.placeholder(dtype=tf.float32,shape=shape,name="in_minValues")
        maxValues=tf.placeholder(dtype=tf.float32,shape=shape,name="in_maxValues")
        minValueWeights=tf.placeholder(dtype=tf.float32,shape=shape,name="in_minValueWeights")
        maxValueWeights=tf.placeholder(dtype=tf.float32,shape=shape,name="in_maxValueWeights")
        boxConstraint=BoxConstraint(minValues,maxValues,minValueWeights,maxValueWeights)
    else:
        boxConstraint=None
    return DataStream(inputTensor,dataType,maskTensor,nCategories=nCategories,ieqConstraints=ieqs,gaussianPrior=gaussianPrior,boxConstraint=boxConstraint,weights=weightsTensor)
 

def stopStreamGradients(streams,gradStopStrength):
    result=[]
    for stream in streams:
        s=stream.copy()
        s.tensor=stopGradient(s.tensor,gradStopStrength)
        if s.logQ is not None:
            s.logQ=stopGradient(s.logQ,gradStopStrength)
        if s.weights is not None:
            s.weights=stopGradient(s.weights,gradStopStrength)
        result.append(s)
    return result

def expandAndReshape(arr,shape):
    if hasattr(arr,"shape"):
        newShape=list(arr.shape)
    else:
        newShape=[]
    while (len(newShape)<len(shape)):
        newShape.insert(0,1)
    return np.broadcast_to(np.reshape(arr,newShape),shape)

def streamFeedDict(stream:DataStream,nSamples=None,feed:DataIn=None):
    """Builds a Tensorflow feed dictionary for a DataStream, using the inputs specified by a DataIn instance.
    Reasonable defaults are substituted for undefined priors etc., allowing flexible usage.
    One must specify either nSamples or the data member of the DataIn argument."""
    assert((nSamples is not None) or ((feed is not None) and feed.data is not None))
    result={}
    if nSamples is None:
        nSamples=feed.data.shape[0]
    else:
        assert((feed is None) or (feed.data is None) or (nSamples==feed.data.shape[0]))
    dataShape=stream.tensor.shape.as_list()
    dataShape[0]=nSamples
    sampleShape=dataShape.copy()
    sampleShape[0]=1
    scalarShape=sampleShape.copy()
    for i in range(len(scalarShape)):
        scalarShape[i]=1

    #Handle the special case where feed is None (unconditional sampling)
    if feed is None:
        result[stream.tensor]=np.zeros(dataShape)
        result[stream.mask]=np.zeros(dataShape)
        result[stream.weights]=np.ones([nSamples,1])
        if stream.gaussianPrior is not None:
            result[stream.gaussianPrior.mean]=np.zeros(sampleShape)
            result[stream.gaussianPrior.sd]=np.zeros(sampleShape)
            result[stream.gaussianPrior.weight]=0
        for i in range(len(stream.ieqConstraints)):
            result[stream.ieqConstraints[i].a]=np.zeros(sampleShape)
            result[stream.ieqConstraints[i].b]=np.zeros(scalarShape)
            result[stream.ieqConstraints[i].weight]=0
        if stream.boxConstraint is not None:
            result[stream.boxConstraint.minValues]=np.zeros(sampleShape)
            result[stream.boxConstraint.maxValues]=np.zeros(sampleShape)
            result[stream.boxConstraint.minValueWeights]=np.zeros(sampleShape)
            result[stream.boxConstraint.maxValueWeights]=np.zeros(sampleShape)
        return result

    #Data tensor
    if feed.data is None:
        result[stream.tensor]=np.zeros(dataShape)
    else:
        result[stream.tensor]=feed.data

    #Mask tensor
    if feed.mask is None:
        result[stream.mask]=np.zeros(dataShape)
    else:
        result[stream.mask]=feed.mask

    #Weights tensor
    if feed.weights is None:
        result[stream.weights]=np.ones([nSamples,1])
    else:
        result[stream.weights]=feed.weights

    #Gaussian prior
    if stream.gaussianPrior is not None:
        if feed.priorMean is None or feed.priorSd is None:
            #Prior not defined: Set placeholder tensors to zero (prior has no effect)
            result[stream.gaussianPrior.mean]=np.zeros(sampleShape)
            result[stream.gaussianPrior.sd]=np.zeros(sampleShape)
            result[stream.gaussianPrior.weight]=0
        else:
            result[stream.gaussianPrior.mean]=expandAndReshape(feed.priorMean,sampleShape) #np.expand_dims(feed.priorMean, axis=0)
            result[stream.gaussianPrior.sd]=expandAndReshape(feed.priorSd,sampleShape) #np.expand_dims(feed.priorSd, axis=0)
            result[stream.gaussianPrior.weight]=1.0 if feed.priorWeight is None else feed.priorWeight
    else:
        #if the stream does not have placeholder tensors for the prior, the feed should not try to specify their values
        assert(feed.priorMean is None and feed.priorSd is None)

    #Inequalities
    if len(stream.ieqConstraints)>0:
        if feed.ieqs is None:
            #Inequalities not defined: Set placeholder tensors to zero (the inequalities have no effect)
            for i in range(len(stream.ieqConstraints)):
                result[stream.ieqConstraints[i].a]=np.zeros(sampleShape)
                result[stream.ieqConstraints[i].b]=np.zeros(scalarShape)
                result[stream.ieqConstraints[i].weight]=0
        else:
            assert(len(stream.ieqConstraints)==len(feed.ieqs))
            for i in range(len(feed.ieqs)):
                assert("a" in feed.ieqs[i])
                assert("b" in feed.ieqs[i])
                aNorm=np.linalg.norm(feed.ieqs[i]["a"])  #to simplify the computations, we make sure that the constraint is specified in normalized form
                result[stream.ieqConstraints[i].a]=expandAndReshape(feed.ieqs[i]["a"]/aNorm,sampleShape) #np.expand_dims(feed.ieqs[i]["a"], axis=0)
                result[stream.ieqConstraints[i].b]=expandAndReshape(feed.ieqs[i]["b"]/aNorm,scalarShape)
                if "weight" in feed.ieqs[i]:
                    result[stream.ieqConstraints[i].weight]=feed.ieqs[i]["weight"]
                else:
                    result[stream.ieqConstraints[i].weight]=1.0

    else:
        #if the stream does not have placeholder tensors for the inequalities, the feed should not try to specify their values
        assert(feed.ieqs is None)

    #Box constraints
    if stream.boxConstraint is not None:
        if feed.minValues is None or feed.maxValues is None:
            #Box constraints not defined: Set placeholder tensors to zero (the box constraints have no effect)
            result[stream.boxConstraint.minValues]=np.zeros(sampleShape)
            result[stream.boxConstraint.maxValues]=np.zeros(sampleShape)
            result[stream.boxConstraint.minValueWeights]=np.zeros(sampleShape)
            result[stream.boxConstraint.maxValueWeights]=np.zeros(sampleShape)
        else:
            assert(feed.minValues is not None)
            assert(feed.maxValues is not None)
            result[stream.boxConstraint.minValues]=expandAndReshape(feed.minValues,sampleShape) #np.expand_dims(feed.minValues, axis=0)
            result[stream.boxConstraint.maxValues]=expandAndReshape(feed.maxValues,sampleShape) # np.expand_dims(feed.maxValues, axis=0)
            if feed.minValueWeights is None:
                result[stream.boxConstraint.minValueWeights]=np.ones(sampleShape)
            else:
                result[stream.boxConstraint.minValueWeights]=expandAndReshape(feed.minValueWeights,sampleShape) #np.expand_dims(feed.minValueWeights, axis=0)
            if feed.maxValueWeights is None:
                result[stream.boxConstraint.maxValueWeights]=np.ones(sampleShape)
            else:
                result[stream.boxConstraint.maxValueWeights]=expandAndReshape(feed.maxValueWeights,sampleShape) #np.expand_dims(feed.maxValueWeights, axis=0)
    else:
        #if the stream does not have placeholder tensors for the box constraints, the feed should not try to specify their values
        assert(feed.minValues is None and feed.maxValues is None)

    return result



#Helper: compute batch averages of data streams
def multiStreamBatchAverage(streams,sampleWeights=None):
    #If no decoder (bwd pass) input data is given, construct the input data streams as batch averages of the encoder output
    result=[]
    for stream in streams:
        if stream.type=="continuous":
            #TODO: properly handle continuous variable masks. For now, we simply assume that no input variables are known.
            averaged=tf.zeros_like(stream.tensor)
            averagedMask=None
            if stream.mask is not None:
                averagedMask=tf.zeros_like(stream.mask)
        else:
            t=streamSoftmax(stream.tensor,stream.nCategories)
            weighted=t if sampleWeights is None else sampleWeights*t
            averaged=tf.reduce_sum(weighted,axis=0,keepdims=True)
            discretePdfNormalize(averaged,nCategories=stream.nCategories)
            averaged=discretePdfLog(averaged,nCategories=stream.nCategories)
            averaged*=tf.ones_like(stream.tensor) #convert to batch
            averagedMask=None
            if stream.mask is not None:
                averagedMask=tf.reduce_mean(stream.mask,axis=0,keepdims=True)*tf.ones_like(stream.mask)
        result.append(DataStream(averaged,stream.type,nCategories=stream.nCategories,mask=averagedMask))
    return result


def discretePdfTruncate(pdf,truncation):
    if truncation is None:
        return pdf
    threshold=truncation*tf.reduce_max(pdf,axis=-1,keepdims=True)
    pdf=tf.nn.relu(tf.sign(pdf-threshold))*pdf
    pdf/=tf.stop_gradient(tf.reduce_sum(pdf,axis=-1,keepdims=True))
    return pdf

def discreteLogPdfTruncate(pdf,truncation):
    if truncation is None:
        return pdf
    threshold=tf.reduce_max(pdf,axis=-1,keepdims=True)+tf.log(truncation)
    pdf=pdf-tf.stop_gradient(1e10*tf.nn.relu(tf.sign(threshold-pdf)))
    return pdf


def logPdfTruncate(pdf,truncation):
    truncated=discretePdfTruncate(tf.nn.softmax(pdf),truncation)
    return tf.log(truncated+1e-20)


def discretePdfApplyTemperature(pdf,temperature):
    if temperature is None:
        return pdf
    if (isinstance(temperature,float) or isinstance(temperature,int)) and temperature==0:
        indices=tf.argmax(pdf,axis=-1)
        return tf.one_hot(indices=indices,depth=pdf.shape[-1].value)
    return tf.nn.softmax(tf.log(pdf+epsilon)/temperature)


#Helper: remove masks from a list of streams
def removeMasks(streams):
    #If no decoder (bwd pass) input data is given, construct the input data streams as batch averages of the encoder output
    result=[]
    for stream in streams:
        result.append(DataStream(stream.tensor,stream.type,None,stream.nCategories))
    return result



#Base class for layers
class Layer:
    def __init__(self):
        #each layer may have trainable variables
        self.variables=[]
        #each layer may or may not contribute to the training loss
        self.loss=0 
        self.bwdLoss=0  #Backward loss currently not used
    #Forward pass. Inputs is a list of DataStream instances, mode is either "training" or "inference"
    def fwd(self,inputs,mode):
        raise NotImplementedError()
    #Forward pass. Data is a list of DataStream instances, mode is either "training" or "sample" or "reconstruct"
    def bwd(self,data,mode):
        raise NotImplementedError()
    #Returns a list of all Tensorflow variables, useful for saving/loading etc.
    def getVariables(self):
        return self.variables


def addTensors(tensorList,streamList):
    assert(len(tensorList)==len(streamList))
    result=[]
    for i in range(len(tensorList)):
        result.append(tensorList[i]+streamList[i].tensor)
    return result

def extractTensors(dataStreamList):
    return [x.tensor for x in dataStreamList]

'''

A manager class for multiple layers. All models inherit from this. 
This class also inherits Layer, which means a LayerStack can manage multiple child stacks.

'''
class LayerStack(Layer):
    def __init__(self,sess=None):
        Layer.__init__(self)
        self.layers=[]
        self.inputs=None
        self.loss=0
        self.bwdLoss=0
        self.initOps=[]
        self.samples=None
        self.layersamples=[]
        self.built=False
        self.fwdLosses=[]
        self.bwdLosses=[]
        self.stageLosses=[]
        self.stageVariables=[]
        self.sess=sess
    #Add a layer
    def add(self,layer):
        self.layers.append(layer)
    #Builds the compute graphs for training, initialization, and inference.
    #This should only be called once on the top-level LayerStack to avoid creating redundant ops.
    def build(self,inputs,bwdSampling=False):
        if type(inputs) is DataStream:
            inputs=[inputs] #everything below expects a list of DataStream instances, but we provide the convenience of calling build() with just a single instance
        self.inputs=inputs
        self.numInputs=len(self.inputs)
        assert(self.built==False)

        #forward and backward passes for training.
        encoded=self.fwd(self.inputs,"training")
        self.trainingInputs=self.layerInputs.copy()  #save for debug visualization
        self.initTrainingStages(self)

        if bwdSampling:
            # Bwd training pass for the bwd bias correction
            # Before it, we batch-average the latent pdf:s and force any remaining residuals to zero, similar to before backward sampling
            averaged=multiStreamBatchAverage(encoded) #,sampleWeights=tf.nn.softmax(-self.layers[-1].inputLogQ))
            for streamIdx in range(self.layers[-1].nNonLatentInputs):
                stream=averaged[streamIdx]
                stream.tensor=tf.zeros_like(stream.tensor)
            self.bwd(data=averaged,mode="training")  

        #data-dependent initialization pass, required before one calls the init() method
        initOutputs=self.fwd(self.inputs,"init")
        for stream in initOutputs:
            self.initOps.append(stream.tensor)

        #Backward sampling
        if bwdSampling:            
            #Sampling fwd and bwd passes
            encoded_bwd=self.fwd(self.inputs,"sample_fwd")
            if sampleInBwdPass:
                averaged = multiStreamBatchAverage(encoded_bwd) #,sampleWeights=tf.nn.softmax(-self.layers[-1].inputLogQ))
            else:
                averaged=encoded_bwd
            for streamIdx in range(self.layers[-1].nNonLatentInputs):
                stream=averaged[streamIdx]
                stream.tensor=tf.zeros_like(stream.tensor)

            self.samples=extractTensors(self.bwd(data=averaged,mode="sample"))

            #To allow visualizing each layer's individual contribution, fill in the layerSample array.
            samples=[0]*self.numInputs #initialize a list of zeros
            self.layerSamples=[]
            for layer in self.layers:
                if hasattr(layer,"xHat_bwd"):
                    samples=addTensors(samples,layer.xHat_bwd[:self.numInputs])
                    assert(not addLastLayerResidualNoise)  #not supported in this bwd sampling
                    self.layerSamples.append(samples)



        #Forward sampling
        else:
            encoded=self.fwd(self.inputs,"sample_fwd")
            #zero out the input residuals:
            for i in range(len(self.inputs)):
                encoded[i].tensor=tf.zeros_like(encoded[i].tensor)
            #Sum the xHat from the forward pass
            samples=[0]*self.numInputs #initialize a list of zeros
            self.layerSamples=[]
            for layer in self.layers:
                if hasattr(layer,"xHat"):
                    samples=addTensors(samples,layer.xHat[:self.numInputs])
                    noisySamples=samples.copy()
                    #add last layer's residual Gaussian noise if needed
                    if addLastLayerResidualNoise:
                        assert(self.numInputs==1 and self.inputs[0].type=="continuous") #the residual noise is only for the paper's visualization, limited to a single continuous input stream
                        residualSd=tf.sqrt(tf.exp(layer.centroidLogVars[0]))
                        residualNoise=tf.random_normal(tf.shape(noisySamples[0]),mean=0,stddev=residualSd)
                        noisySamples[0]+=residualNoise
                    self.layerSamples.append(noisySamples)
            self.samples=self.layerSamples[-1]

        #density estimation based on the previous pass
        self.layerp=[]
        self.layerLogp=[]
        for layer in self.layers:
            if hasattr(layer,"p"):
                self.layerp.append(layer.p)
            if hasattr(layer,"logp"):
                self.layerLogp.append(layer.logp)
        self.p=self.layerp[-1]
        self.logp=self.layerLogp[-1]


        #bookkeeping
        self.built=True

        self.nParameters=0
        for layer in self.layers:
            if hasattr(layer,"nParameters"):
                self.nParameters+=layer.nParameters

    #Initialize the model using a batch of data of shape [nBatch,nVars]
    def init(self,data):
        #for now, this only supports a single input stream
        assert(len(self.inputs)==1)
        if type(data) is list:
            data=data[0]
        self.sess.run(self.initOps,feed_dict={self.inputs[0].tensor:data,self.inputs[0].mask:np.ones_like(data)})

    def fwd(self,inputs,mode):
        if mode=="training":
            self.loss=0
            self.fwdLosses=[]
        self.layerInputs=[]
        for i in range(len(self.layers)):
            layer=self.layers[i]
            print("Fwd pass, layer {}, mode={}".format(layer.__class__.__name__,mode))
            self.layerInputs.append(inputs)
            inputs=layer.fwd(inputs,mode)
            if mode=="training":
                self.loss+=layer.loss
                self.fwdLosses.append(layer.loss)
        return inputs
    def bwd(self,data,mode):
        if mode=="training":
            self.bwdLoss=0
            self.bwdLosses=[]
        for layer in reversed(self.layers):
            print("Bwd pass, layer {}, mode={}".format(layer.__class__.__name__,mode))
            data=layer.bwd(data,mode)
            if mode=="training":
                self.bwdLoss+=layer.bwdLoss
                self.bwdLosses.insert(0,layer.bwdLoss)
        return data
    def getVariables(self):
        result=[]
        for layer in self.layers:
            vars=layer.getVariables()
            if len(vars)>0:
                result.append(vars)
        return list(itertools.chain(*result))
    def initTrainingStages(self,master):
        for layer in self.layers:
            if isinstance(layer,LayerStack):
                layer.initTrainingStages(master)
            elif len(layer.getVariables())>0:
                master.stageLosses.append(layer.loss)
                master.stageVariables.append(layer.getVariables())


#Returns squared distance matrix D with elements d_ij = | a_i - b_j|^2, where a_i = A[i,:] and b_j=B[j,:]
def pairwiseSqDistances(A,B):
    #d_ij=(a_i-b_j)'(a_i-b_j) = a_i'a_i - 2 a_i'b_j + b_j'b_j
    #D = [a_0'a_0, a_1'a_1, ...] - 2 AB' + [b_0'b_0, b_1'b_1, ...]',  assuming broadcasting
    #D = A_d - 2 AB' + B_d
    A_d=tf.reduce_sum(A * A,axis=1,keepdims=True)
    B_d=tf.reshape(tf.reduce_sum(B * B,axis=1),[1,tf.shape(B)[0]])
    return tf.nn.relu(A_d - 2 * tf.matmul(A,B,transpose_b=True) + B_d)  #relu to ensure no negative results due to computational inaccuracy

def maskedPairwiseSqDistances(A,B,mask):
    #* denotes elementwise multiplication, m_i= square root of mask of i:th vector of A, ma_i = shorthand for m_i*a_i
    #d_ij=(m_i*(a_i-b_j))'(m_i*(a_i-b_j)) = ma_i'ma_i - 2 (m_i^2)*a_i'b_j + (m_i*b_j)'(m_i*b_j)
    sqMask=tf.square(mask)
    mA=mask*A
    A_d=tf.reduce_sum(mA*mA,axis=1,keepdims=True)
    B_d=tf.matmul(sqMask,B*B,transpose_b=True)
    return tf.nn.relu(A_d - 2 * tf.matmul(sqMask*A,B,transpose_b=True) + B_d) #relu to ensure no negative results due to computational inaccuracy


#Compute the Mahalanobis distances from samples to Gaussians defined by mean vectors and a global sd vector or scalar
#def sqMahalanobisDistances(samples:tf.Tensor,means:tf.Tensor,sd:tf.Tensor,mask=None):
#    return pairwiseSqDistances(samples/sd,means/sd)
#    if mask is None:
#    else:
#        return maskedPairwiseSqDistances(samples/sd,means/sd,mask)

def sqMahalanobisDistances(samples:tf.Tensor,means:tf.Tensor,sd:tf.Tensor,mask=None):
    if mask is None:
        mask=tf.ones_like(samples)
    mask/=sd
    return maskedPairwiseSqDistances(samples,means,mask)


def interleaveArray(arr:tf.Tensor):
    return tf.reshape(arr,[-1,arr.shape[-1].value])

def deinterleaveArray(arr:tf.Tensor,width,height):
    return tf.reshape(arr,[-1,width,height,arr.shape[-1].value])


class RMM(Layer):
    #inputs : array of [tensor,type] tuples, type = "discrete" or "continuous". For continuous variables, one can also include a mask as the third element
    #masks : array of real-valued tensors. 1 for known variable values, 0 for unknown
    def __init__(self,nCategories:int,nNonLatentInputs=1,allowResample=True,bwdSamplingTemperature=1.0,precisionRho=0.0,inputWidth=1,inputHeight=1,useMaximumLikelihoodLoss=False,gradStopStrength=1.0,trainingPhase=None,isLastLayer=False):
        Layer.__init__(self)

        #Remember the arguments for later
        self.nCategories=nCategories
        self.nNonLatentInputs=nNonLatentInputs
        self.allowResample=allowResample
        self.inputWidth=inputWidth
        self.inputHeight=inputHeight
        self.gradStopStrength=gradStopStrength
        self.useMaximumLikelihoodLoss=useMaximumLikelihoodLoss
        self.isLastLayer=isLastLayer
        self.trainingPhase=trainingPhase
        if trainingPhase is None:
            raise Exception("Training phase argument missing!")

        #bookkeeping: variables will be created later, once we know the input streams
        self.variablesCreated=False
        self.nParameters=0
        self.bwdSamplingTemperature=bwdSamplingTemperature
        self.precisionRho=precisionRho
        self.inputLogQ=0
    def createVariables(self,inputs):
        self.variablesCreated=True
        nInputs=len(inputs)
        self.nInputs=nInputs
        self.centroids=[]
        self.centroidLogVars=[]
        N=self.nCategories  #shorthand

        #Tensorflow variable for marginal class membership probabilities
        if (self.inputWidth>1 or self.inputHeight>1) and arrayMarginalMode=="elementwise":
            #the marginal class distribution is learned for each array element, e.g., image pixel
            self.logMarginalMemberships=tf.Variable(initial_value=np.zeros([1,self.inputWidth,self.inputHeight,self.nCategories]),dtype=tf.float32,trainable=True,name='logMarginalMemberships')
            self.variables.append(self.logMarginalMemberships)
            self.logBwdMarginalMemberships=tf.Variable(initial_value=np.zeros([1,self.inputWidth,self.inputHeight,self.nCategories]),dtype=tf.float32,trainable=True,name='logBwdMarginalMemberships')
            self.variables.append(self.logBwdMarginalMemberships)
            self.bwdVariables=self.logBwdMarginalMemberships
            self.nParameters+=self.inputWidth*self.inputHeight*self.nCategories
        else:
            #the marginal class distribution is global to this layer, i.e., a single categorical PDF of self.nCategories categories
            self.logMarginalMemberships=tf.Variable(initial_value=np.zeros([1,self.nCategories]),dtype=tf.float32,trainable=True,name='logMarginalMemberships')
            self.logBwdMarginalMemberships=tf.Variable(initial_value=np.zeros([1,self.nCategories]),dtype=tf.float32,trainable=True,name='logBwdMarginalMemberships')
            self.variables.append(self.logMarginalMemberships)
            self.variables.append(self.logBwdMarginalMemberships)
            self.bwdVariables=self.logBwdMarginalMemberships
            self.nParameters+=self.nCategories

        self.marginalMemberships=tf.nn.softmax(self.logMarginalMemberships)
        self.logMarginalMemberships=tf.nn.log_softmax(self.logMarginalMemberships)
        self.bwdMarginalMemberships=tf.nn.softmax(self.logBwdMarginalMemberships)
        self.logBwdMarginalMemberships=tf.nn.log_softmax(self.logBwdMarginalMemberships)
        for inputIdx in range(nInputs):
            inputType=inputs[inputIdx].type
            input=inputs[inputIdx].tensor
            M=input.shape[-1].value
            #The trainable variables are N-by-M tensors of N classes parameterized by centroid vectors of M variables.
            varSize=size=[N,M]
            if inputType=="continuous":
                initialValue=np.random.uniform(-1,1,varSize)
                centroids=tf.Variable(initial_value=initialValue,dtype=tf.float32,name='centroids_{}'.format(inputIdx))
                self.nParameters+=M*N
                self.centroids.append(centroids)
                self.variables.append(centroids)
                if sigmaType=="scalar":
                    centroidLogVar=tf.Variable(initial_value=np.log(initialSdScale/np.power(N,1.0/M)),dtype=tf.float32,trainable=True,name='logvar_{}'.format(inputIdx))
                elif sigmaType=="diagonal":
                    centroidLogVar = tf.Variable(initial_value=np.log(initialSdScale / np.power(N, 1.0 / M))*np.ones([1,M]),
                                                 dtype=tf.float32, trainable=True, name='logvar_{}'.format(inputIdx))
                else:
                    raise Exception("unknown sigma type: {}".format(sigmaType))

                self.nParameters+=1
                self.variables.append(centroidLogVar)
                self.centroidLogVars.append(centroidLogVar)

            elif inputType=="discrete":
                initialSpread=0.001
                #Version that stores the centroids in log-space. 
                initialValue=np.random.uniform(-initialSpread,initialSpread,size=varSize)
                centroids=tf.Variable(initial_value=initialValue,dtype=tf.float32,name='centroids_{}'.format(inputIdx))
                self.nParameters+=M*N
                self.variables.append(centroids)
                centroids=streamLogSoftmax(centroids,inputs[inputIdx].nCategories)
                self.centroids.append(centroids)
                self.centroidLogVars.append(None)

            else:
                raise Exception("Invalid input type")
    def fwd(self,inputs,mode):
        inputs=stopStreamGradients(inputs,self.gradStopStrength)
        #Some input checks
        for stream in inputs:
            assert(len(stream.tensor.shape)==len(inputs[0].tensor.shape))  #all streams must have the same shape dimensionality (e.g., can't mix images and vectors)
            assert(len(stream.tensor.shape)==2)


        #create TensorFlow variables that are shared between all passes
        #(we could not create these in the constructor, as we need to know the input streams first)
        if not self.variablesCreated:
            self.createVariables(inputs)

        #some helpers
        batchSize=tf.shape(inputs[0].tensor)[0] 

        N=self.nCategories
        nInputs=self.nInputs
        self.inputs=inputs
        sampleWeights=inputs[-1].weights  #convention: sample weights are carried with the last input stream so that weights updated by this layer can be passed on with the new stream added by this layer's latent
        if sampleWeights is None:
            sampleWeights=tf.ones([batchSize,1])

        #Randomize input sample indices for data-dependent initialization
        if mode=="init":
            idxs = tf.range(batchSize)
            ridxs = tf.random_shuffle(idxs)[:N]

        #The q(h) of previous layers is passed along the last input stream
        if inputs[-1].logQ is None:
            logQ=tf.zeros([batchSize,1])
        else:
            logQ=stopGradient(inputs[-1].logQ,self.gradStopStrength)

        #Compute log memberships based on latent and non-latent variable inputs
        membershipShape=[batchSize,N]
        self.nonLatentLogMemberships=tf.zeros(membershipShape)     #first init to zero
        self.latentLogMemberships=tf.zeros(membershipShape)     #first init to zero
        initialCentroids=[]                             #placeholder
        self.inputInfoSum=tf.zeros([batchSize,1])
        for inputIdx in range(nInputs):
            input=inputs[inputIdx].tensor
            input=input
            inputType=inputs[inputIdx].type
            centroids=self.centroids[inputIdx]
            mask=inputs[inputIdx].mask
            mask=tf.stop_gradient(mask)
            if inputIdx<self.nNonLatentInputs:
                #inputInfoSum: this will be zero for inputs with no known value and no priors or constraints, non-zero otherwise 
                #this is used in determining the encoder-decoder skip-connection weights
                self.inputInfoSum+=tf.reduce_sum(mask,axis=-1,keepdims=True)
                if inputs[inputIdx].gaussianPrior is not None:
                    self.inputInfoSum+=inputs[inputIdx].gaussianPrior.weight
                for i in range(len(inputs[inputIdx].ieqConstraints)):
                    self.inputInfoSum+=inputs[inputIdx].ieqConstraints[i].weight
                if inputs[inputIdx].boxConstraint is not None:
                    self.inputInfoSum+=tf.reduce_sum(inputs[inputIdx].boxConstraint.minValueWeights,axis=-1,keepdims=True)
                    self.inputInfoSum+=tf.reduce_sum(inputs[inputIdx].boxConstraint.maxValueWeights,axis=-1,keepdims=True)

            M=input.shape[-1].value
            if inputType=="continuous":
                centroidLogVar=self.centroidLogVars[inputIdx]

                #data-dependent initialization
                if mode=="init":
                    inputToGatherFrom=input
                    inputMean=tf.reduce_mean(inputToGatherFrom,axis=0,keepdims=True)
                    inputVar=tf.reduce_mean(tf.square(inputToGatherFrom-inputMean),axis=0,keepdims=True)
                    inputSd=tf.sqrt(inputVar)
                    if dataDependentInitializationMode=="select":
                        #Assign class prototypes to randomly selected input samples.
                        #Also, add some noise to prevent duplicates in case the same input is selected multiple times.
                        initCentroids = tf.gather(inputToGatherFrom, ridxs)
                        initCentroids+=tf.truncated_normal(shape=[N,M],mean=tf.zeros(M),stddev=0.1*inputSd)
                    elif dataDependentInitializationMode=="normal":
                        #Sample prototypes randomly from a diagonal Gaussian approximation of the input data
                        initCentroids=tf.truncated_normal(shape=[N,M],mean=inputMean[0],stddev=inputSd[0])
                    elif dataDependentInitializationMode=="uniform":
                        inputMin=tf.reduce_min(inputToGatherFrom,axis=0,keepdims=True)
                        inputMax=tf.reduce_max(inputToGatherFrom,axis=0,keepdims=True)
                        initCentroids=tf.random_uniform(shape=[N,M],minval=inputMin,maxval=inputMax)
                    else:
                        raise Exception("Invalid data dependent initialization mode: {}".format(dataDependentInitializationMode))
                    centroids=tf.assign(centroids,initCentroids)
                    initialCentroids.append(centroids)
                    #Init centroid stdevs proportional to average distances between the centroids, assuming uniform distribution
                    #With 1D data, we simply divide the sd of data by the number of classes. In higher dimensions,
                    #the distance decreases much more slowly with the number of classes, as more centroids are needed to fill the space.
                    if sigmaType=="scalar":
                        inputVar = tf.reduce_mean(inputVar)
                        inputSd = tf.sqrt(inputVar)
                        initSd=(initialSdScale*inputSd)/np.power(N,1.0/M)
                    elif sigmaType=="diagonal":
                        initSd=initialSdScale * inputSd
                    else:
                        raise Exception("Unknown sigma type: {}".format(sigmaType))
                    centroidLogvar=tf.assign(centroidLogVar,tf.log(tf.square(initSd)))

                #The membership computation needs both log variances and variances
                centroidSd=tf.exp(0.5*centroidLogVar)
                centroidVar=tf.exp(centroidLogVar)

                #membership log-probabilities, resulting in an batchSize-by-N tensor
                #logp of diagonal gaussian: -0.5*[tf.square(x-mean)/var+logVar]
                streamLogMemberships=-0.5*(sqMahalanobisDistances(input,centroids,centroidSd,mask)+tf.reduce_sum(mask*centroidLogVar,axis=-1,keepdims=True))

                '''
                Handle inequality constraints of type a'x+b > 0,
                i.e., defining a half-space of valid x on one side of a constraint hyperplane.
                We integrate the Gaussian pdf of each mixture component over the half-space,
                and use that as a membership probability multiplier. In other words, the multiplier equals the
                probability mass within the valid half-space
                Simplification: As each component is isotropic, and a'x+b is the distance from the hyperplane (we assume norm(a)=1, which is ensured by streamFeedDict()),
                positive distances indicating valid x, we only need to compute p=(a'mu+b)/sigma and integrate a
                standard 1D Gaussian from -inf to p, which equals the CDF of standard Gaussian at p, i.e.,
                0.5*(1+erf(p/sqrt(2))
                '''
                if mode=="sample_fwd": #only consider the constraints during inference
                    for c in inputs[inputIdx].ieqConstraints:
                        if sigmaType!="scalar":
                            print("WARNING: use of the a'x+b>0 inequalities is inaccurate with non-scalar sigma")
                        p=(tf.tensordot(c.a,centroids,[-1,-1])+c.b)/tf.sqrt(tf.reduce_mean(centroidVar))  #take mean to approximate if non-scalar sigma
                        cdf=0.5*(1.0+tf.erf(p/tf.sqrt(2.0)))
                        streamLogMemberships+=c.weight*tf.log(cdf+epsilon)


                '''
                Handle box constraints, i.e., simplified per-variable inequality constraints
                '''
                if mode=="sample_fwd": #only consider the constraints during inference
                    bc=inputs[inputIdx].boxConstraint
                    if bc is not None:
                        #compute a [nSamples,nComponents,nVars] tensor of maxValue-centroidValue,
                        #from the [nSamples,nVars] and [nComponents,nVars] tensors of maxValues and component means
                        #(recall that N=nComponents, M=nVars) 
                        expandedValues=tf.reshape(bc.maxValues,[-1,1,M])
                        expandedCentroids=tf.reshape(centroids,[1,N,M])
                        p=(expandedValues-expandedCentroids)/centroidSd
                        #evaluate the Gaussian CDFs to get the probability mass on the correct side of the constraint
                        cdf=0.5*(1.0+tf.erf(p/tf.sqrt(2.0)))
                        #sum the log-probabilities of each variable, accumulate the log memberships
                        streamLogMemberships+=tf.reduce_sum(tf.reshape(bc.maxValueWeights,[-1,1,M])*tf.log(cdf+epsilon),axis=-1)

                        #the same for min values
                        expandedValues=tf.reshape(bc.minValues,[-1,1,M])
                        expandedCentroids=tf.reshape(centroids,[1,N,M])
                        p=(expandedCentroids-expandedValues)/centroidSd
                        cdf=0.5*(1.0+tf.erf(p/tf.sqrt(2.0)))
                        streamLogMemberships+=tf.reduce_sum(tf.reshape(bc.minValueWeights,[-1,1,M])*tf.log(cdf+epsilon),axis=-1)

                '''
                Handle the optional Gaussian prior. For each mixture component, we compute the 
                integral of the component PDF multiplied with the prior, and use that as a membership probability
                scaling factor. 

                We compute this scaling factor separately for each variable, assuming diagonal prior covariance.
                The scale factors combine multiplicatively, i.e., we can add the logs.
               
                Equation given in: http://luispedro.org/files/derivations/gaussian_integral.pdf (computer checked):

                integrate(N(x | mu1,sigma1^2)N(x | mu2,sigma2^2))=N(mu1 | mu2, sigma1^2+sigma2^2)
                '''
                if mode=="sample_fwd": #only consider the constraints during inference
                    prior=inputs[inputIdx].gaussianPrior
                    if prior is not None:
                        #assert(prior.sd.shape[0].value==1 and prior.sd.shape[1].value==M)
                        varSum=centroidVar+tf.square(prior.sd)
                        #The full equation: 1.0-tf.reduce_sum(0.5*tf.log(2*np.pi*varSum))-0.5*(sqMahalanobisDistances(prior.mean,centroids,tf.sqrt(varSum)))
                        #However, we can omit the first terms, as they are same for all input vectors and centroids
                        priorLogMemberships=-0.5*(sqMahalanobisDistances(prior.mean,centroids,tf.sqrt(varSum)))
                        streamLogMemberships+=prior.weight*priorLogMemberships

            elif inputType=="discrete":
                if mode=="init":
                    initialCentroids.append(centroids)  #no data-driven init for discrete data streams

                #Dirichlet, assuming both inputs and centroids as pdf:s:
                #log p(input | centroid)=-log(beta(centroid)) + sum ((centroid-1)*log(input))
                #Because inputs are logits and centroids are log-pdf:s
                #log p(input | centroid)=-log(beta(softmax(centroid)) + sum((softmax(centroid)-1)*log_softmax(input)
                #input=streamLogSoftmax(input,inputs[inputIdx].nCategories)
                #centroids=streamSoftmax(centroids,inputs[inputIdx].nCategories)
                #masked=mask*input
                #streamLogMemberships=tf.tensordot(masked,centroids-1.0,[1,1])
                #streamLogMemberships-=tf.reshape(tf.lbeta(centroids),[1,-1]) #the beta does not depend on input => constant for each component

                #Categorical, assuming log-one-hot input and log-pdf centroids:
                # log p(input | centroid) = softmax(input)*centroid
                input=streamSoftmax(input,inputs[inputIdx].nCategories)
                masked=mask*input
                streamLogMemberships=tf.tensordot(masked,centroids,[1,1])

                #Multinomial, assuming log-pdf centroids and inputs:
                # log p(input | centroid) = -log(beta(x+1) + softmax(input)*centroid
                #input=streamSoftmax(input,inputs[inputIdx].nCategories)
                #masked=mask*input
                #streamLogMemberships=tf.tensordot(masked,centroids,[1,1])
                #if mode!="training" or includeMultinomialBetaDuringTraining:
                #    streamLogMemberships-=tf.reshape(tf.lbeta(masked+1.0),[-1,1])

            else:
                raise Exception("Invalid input type")
            #Sample precision control
            if inputIdx>=self.nNonLatentInputs:
                self.latentLogMemberships+=streamLogMemberships
            else:
                self.nonLatentLogMemberships+=streamLogMemberships

        #Expand the marginal memberships to same shape as the batch if needed
        if (self.inputWidth>1 or self.inputHeight>1) and arrayMarginalMode=="elementwise":
            #The marginal memberships are of shape [1,width,height,nCategories].
            #The log-memberships computed above are of shape [-1,nCategories], with array elements interleaved.
            #To allow adding the marginals to the We must expand the marginals to [nBatch,width,height,nCategories],
            #and then flatten to [-1,nCategories]
            arrayBatchSize=batchSize//(self.inputWidth*self.inputHeight)
            logMarginalMemberships=tf.broadcast_to(self.logMarginalMemberships,[arrayBatchSize,self.inputWidth,self.inputHeight,self.nCategories])
            logMarginalMemberships=tf.reshape(logMarginalMemberships,[-1,self.nCategories])
        else:
            logMarginalMemberships=self.logMarginalMemberships

            
        #Log-memberships, including modified log-memberships for training E-step, M-step, and the U-net skips
        if useCurriculum and (mode=="training" or mode=="init"):
            marginalWeight=tf.nn.relu(tf.sign(self.trainingPhase-2.0/3.0))
            latentWeight=tf.nn.relu(tf.sign(self.trainingPhase-1.0/3.0))
        else:
            marginalWeight=1.0
            latentWeight=1.0

        self.mStepLogMemberships=self.nonLatentLogMemberships+marginalWeight*logMarginalMemberships+latentWeight*self.latentLogMemberships

        #self.mStepLogMemberships=self.nonLatentLogMemberships+logMarginalMemberships+StepPrecision*self.latentLogMemberships
        if modifiedEStep:
            self.eStepLogMemberships=self.nonLatentLogMemberships+self.precisionRho*self.latentLogMemberships
        else:
            self.eStepLogMemberships=self.mStepLogMemberships
        
        #self.logMemberships will hold the logits that are passed on to the next layer
        self.logMemberships=self.mStepLogMemberships 


        #density estimation
        def stableMixtureLogDensity(logM):           
            #sum(exp(logm1+shift)+exp(logm2+shift)+...)=exp(shift)*sum(exp(logm1)+exp(logm2)+...)
            #=>log(sum(exp(logm1)+exp(logm2)+...))=log(sum(exp(logm1+shift)+exp(logm2+shift)+...))-shift
            logShift=-tf.stop_gradient(tf.reduce_max(logM,axis=-1,keepdims=True)) #to ensure no underfloaws
            p=tf.reduce_sum(tf.exp(logM+logShift),axis=-1,keepdims=True) 
            return tf.log(p)-logShift   
        self.logp=stableMixtureLogDensity(self.mStepLogMemberships)
        self.logp+=tf.log(sampleWeights)
        self.layerLogp=self.logp
        self.logp-=logQ
        self.p=tf.exp(self.logp)

        #likelihood estimation only based on non-latent inputs (for DRMMBlockHierarchy, which discards the non-latent input stream residuals, assuming that they contain no information)
        self.nonLatentLogP=stableMixtureLogDensity(self.nonLatentLogMemberships+logMarginalMemberships)
        self.nonLatentLogP+=tf.log(sampleWeights)
        self.nonLatentLogP-=logQ


        #Training losses
        if mode=="training":
            eStepIndices=tf.argmax(self.eStepLogMemberships,axis=-1)
            self.eStepMemberships=tf.one_hot(indices=eStepIndices,depth=self.eStepLogMemberships.shape[-1].value)
            EMLosses= -tf.reduce_sum(tf.stop_gradient(self.eStepMemberships)*self.mStepLogMemberships,axis=-1)
            EMLoss=tf.reduce_mean(sampleWeights[:,0]*EMLosses)

            #Regularization loss: similar to above, but treating centroids as data, and data as centroids.
            #This prevents orphan centroids, as a centroid being very far from all data points causes a large loss.
            #Obviously, we assume large enough minibatch sizes such that the batches are representative.
            self.regularizationLoss=-regularization*tf.reduce_mean(tf.reduce_max(self.nonLatentLogMemberships,axis=0)) 

            #Total loss
            #decayingPhase=tf.clip_by_value(tf.square(2.0-2.0*self.trainingPhase),0,1)
            decayingPhase=1.0-self.trainingPhase
            self.loss=EMLoss+decayingPhase*self.regularizationLoss
            if useCurriculum:
                useRegularization=tf.nn.relu(tf.sign(2.0/3.0-self.trainingPhase))
            else:
                useRegularization=0.0
            proxyLoss=-tf.reduce_mean(self.logp)+useRegularization*self.regularizationLoss
            if self.isLastLayer:
                trueLoss=-tf.reduce_mean(self.logp)
            else:
                trueLoss = 0.0
            self.maximumLikelihoodLoss=trueLoss
            #trueLossMix=1.0-isFirstStage
            if useCurriculum:
                trueLossMix=1.0-tf.clip_by_value(3.0*decayingPhase,0,1)
                self.loss=trueLossMix*trueLoss+(1.0-trueLossMix)*proxyLoss
            else:
                self.loss=trueLoss
            #
            # self.maximumLikelihoodLoss=-tf.reduce_mean(self.logp)+isFirstStage*self.regularizationLoss
            # #if self.useMaximumLikelihoodLoss:
            # self.loss=self.maximumLikelihoodLoss
            # #else:
            # #    mlMix=1.0-self.gradStopStrength
            # #    self.loss = (1.0 - mlMix) * self.loss + mlMix * self.maximumLikelihoodLoss
            # if not self.isLastLayer:
            #     #When gradients are enabled, we only use the last layer's loss, as that's the true EM loss.
            #     #The sum of all layer losses is just a proxy that enables better initial convergence
            #     #self.loss*=decayingPhase
            #     self.loss*=tf.clip_by_value(3.0*decayingPhase,0,1)
            #     #self.loss*=self.gradStopStrength

        # Save the membership pdf for the fwd-bwd skip connection of DRMM patch hierarchies for sequential/array data
        self.skipLogMemberships=self.logMemberships
        self.skipMemberships=tf.nn.softmax(self.logMemberships)

        #Truncate (optional) and log-softmax to get final log-membership probabilities
        if mode=="training":
            if trainingTemperature is None:
                self.logMemberships=tf.log(tf.one_hot(indices=tf.argmax(self.logMemberships,axis=-1),depth=self.logMemberships.shape[-1].value))
            else:
                self.logMemberships/=trainingTemperature
            self.logMemberships = discreteLogPdfTruncate(self.logMemberships, trainingFwdTruncation)
        else:
            self.logMemberships=discreteLogPdfTruncate(self.logMemberships,fwdTruncation)
        self.logMemberships=tf.nn.log_softmax(self.logMemberships)

        #Convert membership logits to probabilities
        self.memberships=tf.nn.softmax(self.logMemberships)
        #if mode!="training":
        #    self.memberships=discretePdfTruncate(self.memberships,fwdTruncation)
        #onehotMemberships = tf.one_hot(indices=tf.argmax(self.memberships, axis=-1), depth=self.memberships.shape[-1].value)
        #useOnehot=tf.nn.relu(tf.sign(self.trainingPhase-2.0/3.0))
        #self.memberships=useOnehot*onehotMemberships+(1.0-useOnehot)*self.memberships


        #Determine the layer's latent variable (component membership per input vector) by sampling or taking argmax
        self.inputLogQ = logQ  # keep a copy of the pre-update logQ for later
        # if (mode=="training" or mode=="init") and (teacherForcing or (GumbelTemperature>0.0)):
        #     if mode=="init":
        #         #Repeating the training code above. TODO: reorganize to remove redundancy
        #         if GumbelTemperature > 0.0:
        #             self.eStepMemberships = gumbel_softmax_sample(self.eStepLogMemberships, EMtemperature)
        #         else:
        #             eStepIndices = tf.argmax(self.eStepLogMemberships, axis=-1)
        #             self.eStepMemberships = tf.one_hot(indices=eStepIndices,
        #                                                depth=self.eStepLogMemberships.shape[-1].value)
        #
        #     #Interpreting the EM above as supervised training with argmax of self.eStepMemberships as the targets,
        #     #we output the targets instead of sampling
        #     self.memberships=self.eStepMemberships
        #     if GumbelTemperature>0.0:
        #         xHat = self.reconstruct(self.eStepMemberships, centroids=initialCentroids if mode == "init" else None)
        #     else:
        #         xHat = self.reconstruct_indices(eStepIndices, centroids=initialCentroids if mode == "init" else None)
        # else:
        if reparameterizedSampling and mode=="training":
            self.memberships=sampleReparameterized(self.logMemberships)
            #probs=tf.nn.softmax(self.logMemberships)
            logQ+=tf.reduce_sum(tf.nn.log_softmax(self.logMemberships)*tf.stop_gradient(self.memberships),axis=-1,keepdims=True)
            indices = tf.argmax(self.memberships, axis=-1)
        else:
            probs = self.memberships
            distr=tf.distributions.Categorical(probs=probs)
            indices=distr.sample()
            self.memberships=tf.one_hot(indices=indices,depth=self.memberships.shape[-1].value)

            #update importance weights that will be passed to the next layer
            logQ += tf.reduce_sum(self.logMemberships * self.memberships, axis=-1, keepdims=True)

        #Compute reconstructed inputs and reconstruction residuals and losses.
        xHat=self.reconstruct_indices(indices,centroids=initialCentroids if mode=="init" else None)
        self.xHat=xHat
        r=self.residual(inputs,xHat)

        #Everything done, construct output streams array
        outputs=r

        #The latent output mask is always 1, as uncertainty is captured by the distribution of the sampled latent variables.
        outputMask=tf.ones(membershipShape)
        self.outputMask=outputMask
        outputs.append(DataStream(discretePdfLog(self.memberships,self.nCategories),"discrete",nCategories=self.nCategories,mask=outputMask,weights=sampleWeights,logQ=logQ))
        return stopStreamGradients(outputs,self.gradStopStrength)

    def meanInputSd(self):
        result=0
        nContinuous=0
        for inputIdx in range(self.nInputs):
            inputType=self.inputs[inputIdx].type
            if inputType=="continuous":
                nContinuous+=1
                result+=tf.exp(self.centroidLogVars[inputIdx])
        if nContinuous==0:
            return 1.0
        return tf.sqrt(result/nContinuous)

    def bwd(self,data,mode):     
        logBwdMemberships=data[-1].tensor
        arrayBatchSize=tf.shape(logBwdMemberships)[0]//(self.inputWidth*self.inputHeight)
        #Sample the memberships if needed
        if mode=="reconstruct_onehot":
            indices=tf.argmax(logBwdMemberships,axis=-1)
        else:
            if (not sampleInBwdPass):
                #Determine whether we should incorporate information from the forward pass
                #skipWeight=UnetSkipStrength*self.inputInfoSum/(tf.reduce_max(self.inputInfoSum)+epsilon)
                skipWeight = UnetSkipStrength * tf.sign(self.inputInfoSum)
                # Bwd pass is doing simple reconstruction by always picking h=argmax(logBwdMemberships)
                # => we can simply add the fwd pass log-memberships so that if both fwd and bwd memberships agree, this has no effect,
                # but if they disagree, the fwd memberships can nudge us to the correct direction.
                logBwdMemberships += skipWeight * self.skipLogMemberships
                indices = tf.argmax(logBwdMemberships, axis=-1)
            else:
                #U-net -style encoder-decoder skip-connections. The skip connection weight is set to zero if all the encoder inputs are unknown (i.e., sign(sum(masks))==0)
                if (UnetSkipStrength is not None) and mode!="training":
                    skipWeight=UnetSkipStrength*tf.sign(self.inputInfoSum)
                    #Final fwd pass samples were averaged before sampling in bwd pass => need to similarly average the fwd-bwd skip connection
                    if self.inputWidth>1 or self.inputHeight>1:
                        #If the data is interleaved arrays, we need to deinterleave the skip-connection memberships before batch-averaging
                        skipMemberships=deinterleaveArray(self.skipMemberships,self.inputWidth,self.inputHeight)
                    else:
                        skipMemberships=self.skipMemberships
                    skipMemberships=discretePdfBatchAverage(skipMemberships,nCategories=self.nCategories) #,sampleWeights=tf.nn.softmax(-self.inputLogQ))
                    if self.inputWidth>1 or self.inputHeight>1:
                        #Undo the deinterleaving above
                        skipMemberships*=tf.ones([arrayBatchSize,self.inputWidth,self.inputHeight,self.nCategories])  #First broadcast to undo the shape change due to averaging
                        skipMemberships=interleaveArray(skipMemberships)
                    logBwdMemberships=(1.0-skipWeight)*logBwdMemberships+skipWeight*discretePdfLog(skipMemberships,self.nCategories)
                #Learn and apply the bias correction. The correction is multiplied by sign of bwdSamplingTemperature so that
                #it is not applied for layers that do not sample (temperature 0)
                #If we do not process arrays or if arrayMarginalMode!="elementWise", things are simple:
                if not((self.inputWidth>1 or self.inputHeight>1) and arrayMarginalMode=="elementwise"):
                    logBwdMemberships=self.logBwdMarginalMemberships + tf.stop_gradient(logBwdMemberships)
                    if mode=="training":
                        #Bias correction loss: batch-averages of bwd memberships and fwd memberships should match
                        targets=tf.stop_gradient(discretePdfBatchAverage(self.skipMemberships,self.nCategories))
                        logits=tf.log(discretePdfBatchAverage(tf.nn.softmax(logBwdMemberships),nCategories=self.nCategories))
                        self.bwdLoss=-tf.reduce_mean(tf.reduce_sum(targets*logits,axis=-1))
                #Otherwise, we have to do a bit of interleaving and deinterleaving
                else:
                    logBwdMemberships=deinterleaveArray(logBwdMemberships,self.inputWidth,self.inputHeight)
                    logBwdMemberships=self.logBwdMarginalMemberships + tf.stop_gradient(logBwdMemberships)
                    if mode=="training":
                        #Bias correction loss: batch-averages of bwd memberships and fwd memberships should match
                        targets=tf.stop_gradient(discretePdfBatchAverage(deinterleaveArray(self.skipMemberships,self.inputWidth,self.inputHeight),self.nCategories))
                        logits=tf.log(discretePdfBatchAverage(tf.nn.softmax(logBwdMemberships),nCategories=self.nCategories))
                        self.bwdLoss=-tf.reduce_mean(tf.reduce_sum(targets*logits,axis=-1))
                    logBwdMemberships=interleaveArray(logBwdMemberships)

                #Truncate
                logBwdMemberships=logPdfTruncate(logBwdMemberships,bwdTruncation)

                #Sample. Always use temperature = 1 when training to learn the bias correction, but allow temperature control for inference
                probs=softmaxWithTemperature(logBwdMemberships,1.0 if mode=="training" else self.bwdSamplingTemperature)
                distr=tf.distributions.Categorical(probs=probs)
                indices=distr.sample()

        #Reconstruct this layer's input
        xHat=self.reconstruct_indices(indices)  #stop gradients to prevent the bwd loss from affecting fwd pass parameters
        self.xHat_bwd=xHat

        #Combine with residual
        x=self.residual_inverse(xHat,data[:-1])
        return x
    def reconstruct(self,memberships,centroids=None):
        if centroids is None:
            centroids=self.centroids
        reconstructed_list=[]
        #assert(self.memberships.shape.as_list()==memberships.shape.as_list())
        for inputIdx in range(self.nInputs):
            inputType=self.inputs[inputIdx].type
            if inputType=="continuous":
                reconstructed=tf.matmul(memberships,centroids[inputIdx])
            else:
                reconstructed=tf.matmul(memberships,streamSoftmax(centroids[inputIdx],self.inputs[inputIdx].nCategories))
                reconstructed/=tf.reduce_sum(reconstructed,axis=-1,keepdims=True)
                reconstructed=discretePdfLog(reconstructed,self.inputs[inputIdx].nCategories)

            reconstructed_list.append(DataStream(reconstructed,inputType,nCategories=self.inputs[inputIdx].nCategories))
        return reconstructed_list
    def reconstruct_indices(self,membershipIndices,centroids=None):
        if centroids is None:
            centroids=self.centroids
        reconstructed_list=[]
        #assert(self.memberships.shape.as_list()==memberships.shape.as_list())
        for inputIdx in range(self.nInputs):
            inputType=self.inputs[inputIdx].type
            reconstructed=tf.gather(centroids[inputIdx],membershipIndices)

            reconstructed_list.append(DataStream(reconstructed,inputType,nCategories=self.inputs[inputIdx].nCategories))
        return reconstructed_list

    def residual(self,x,xHat):
        residuals_list=[]
        for inputIdx in range(len(x)):
            input=x[inputIdx].tensor
            inputType=x[inputIdx].type
            rIeqConstraints=None
            rGaussianPrior=None
            rBoxConstraint=None
            if inputType=="continuous":
                r=input-xHat[inputIdx].tensor
                #If input stream has inequality constraints of form a'x+b>0,
                #these must be transformed into corresponding residual constraints a_r'r+b_r>0.
                #As r=x-xHat, x=r+xHat. Substituting this in the constraint inequality, we get a'(r+xHat)+b=a'r+(b+a'xHat)>0.
                #Thus, a_r=a, b_r=b+a'xHat
                if x[inputIdx].ieqConstraints is not None:
                    rIeqConstraints=[]
                    for c in x[inputIdx].ieqConstraints:
                        a_r=c.a
                        aTxHat=tf.reduce_sum(c.a*xHat[inputIdx].tensor,axis=-1,keepdims=True) #this computes the dot product separately for each a and xHat in the batch
                        b_r=c.b+aTxHat
                        rIeqConstraints.append(IEQConstraint(a=a_r,b=b_r,weight=c.weight))
                #Gaussian priors get shifted similar to input vectors
                if x[inputIdx].gaussianPrior is not None:
                    rGaussianPrior=GaussianPrior(mean=x[inputIdx].gaussianPrior.mean-xHat[inputIdx].tensor,sd=x[inputIdx].gaussianPrior.sd,weight=x[inputIdx].gaussianPrior.weight)

                #Box constraints get shifted similar to input vectors
                bc=x[inputIdx].boxConstraint
                if bc is not None:
                    rBoxConstraint=BoxConstraint(minValues=bc.minValues-xHat[inputIdx].tensor,maxValues=bc.maxValues-xHat[inputIdx].tensor,minValueWeights=bc.minValueWeights,maxValueWeights=bc.maxValueWeights)
            elif inputType=="discrete":
                #r=input
                r=input-xHat[inputIdx].tensor
                r=streamLogSoftmax(r,nCategories=x[inputIdx].nCategories)
            else:
                raise Exception("Invalid input type")
            residuals_list.append(DataStream(r,inputType,x[inputIdx].mask,x[inputIdx].nCategories,ieqConstraints=rIeqConstraints,gaussianPrior=rGaussianPrior,boxConstraint=rBoxConstraint,weights=x[inputIdx].weights))
        return residuals_list

    def residual_inverse(self,xHat,r):
        x_list=[]
        for inputIdx in range(self.nInputs):
            inputType=self.inputs[inputIdx].type 
            rt=r[inputIdx].tensor
            if inputType=="continuous":
                x=xHat[inputIdx].tensor+rt
            elif inputType=="discrete":
                x=xHat[inputIdx].tensor+rt
                x=streamLogSoftmax(x,nCategories=xHat[inputIdx].nCategories)
            else:
                raise Exception("Invalid input type")
            x_list.append(DataStream(x,inputType,mask=None,nCategories=xHat[inputIdx].nCategories))
        return x_list

 
class DRMM(LayerStack):
    def __init__(self,sess,nLayers,nComponentsPerLayer,inputs,initialLearningRate=0.001,finalEStepPrecision=1.0,useBwdSampling=False,train=True,initialEStepPrecision=0.0,endToEnd=True):
        self.sess=sess
        if type(inputs) is DataStream:
            inputs=[inputs]  #everything below expects a list of DataStream objects     
        self.inputs=inputs.copy()
        LayerStack.__init__(self,sess)
        self.useBwdSampling=useBwdSampling
        useMaximumLikelihoodLoss=False #endToEnd
        if train:
            self.trainingPhase=tf.Variable(dtype=tf.float32,initial_value=1.0,trainable=False)
            thresholdedPhase=tf.nn.relu(tf.sign(self.trainingPhase-1.0/3.0))
            thresholdedPhase2=tf.nn.relu(tf.sign(self.trainingPhase-2.0/3.0))
            self.precisionRho=initialEStepPrecision+(finalEStepPrecision-initialEStepPrecision)*thresholdedPhase
            decayingPhase = tf.clip_by_value(1.0-3.0*(self.trainingPhase - 2.0 / 3.0),0.0,1.0)   # linearly decay from 1.0 to zero in the 3rd phase
            decayingPhase*=tf.clip_by_value(1.0-thresholdedPhase2,0.1,1.0)  # linearly decay from 0.1 to zero in the 3rd phase
            #decayingPhase=1.0-self.trainingPhase
            if useCurriculum:
                self.learningRate=initialLearningRate*decayingPhase  #first keep constant, start decaying quadratically in the middle
            else:
                self.learningRate=tf.constant(initialLearningRate,dtype=tf.float32)
            self.thresholdedPhase=thresholdedPhase
            #gradStopStrength=1.0-thresholdedPhase2 if endToEnd else 1.0
            if useCurriculum:
                gradStopStrength = 1.0 - thresholdedPhase2 if endToEnd else 1.0
            else:
                gradStopStrength=0.0
        for i in range(nLayers):
            self.add(RMM(nComponentsPerLayer,
                         nNonLatentInputs=len(inputs),
                         precisionRho=self.precisionRho if train else finalEStepPrecision,
                         useMaximumLikelihoodLoss=useMaximumLikelihoodLoss,
                         gradStopStrength=gradStopStrength,
                         trainingPhase=self.trainingPhase,
                         isLastLayer=i==nLayers-1))
        self.build(inputs,bwdSampling=useBwdSampling)
        if sampleInBwdPass:
            self.loss+=self.bwdLoss
        if train:
            self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learningRate)
            # gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            # gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
            # optimizeOp = self.optimizer.apply_gradients(zip(gradients, variables))
            # self.optimize=optimizeOp
            # self.pretrain=optimizeOp
            self.optimize=self.optimizer.minimize(self.loss)
    def train(self,phase,dataBatch,weights=None):
        self.trainingPhase.load(phase,self.sess)
        feed_dict={}
        if not isinstance(dataBatch,list):
            dataBatch=[dataBatch]
        for i in range(len(self.inputs)):
            feed_dict[self.inputs[i].tensor]=dataBatch[i]
            feed_dict[self.inputs[i].mask]=np.ones_like(dataBatch[i])
            feed_dict[self.inputs[i].weights]=np.ones([dataBatch[i].shape[0],1]) if weights is None else weights
        loss,lr,rho,temp=self.sess.run([self.loss,self.learningRate,self.precisionRho,self.optimize],feed_dict)
        return {"loss":loss,"lr":lr,"rho":rho,"logp":0.0} #logp output not supported anymore, but kept to not break any training code that logs it
    def sample(self,nSamples=None,inputs=None,getLogp=False,sorted=True,sampledLayer=None):
        """
        Generate a batch ofsamples.

        Parameters:
        inputs      DataIn instance or a list of DataIn instances if this model has multiple input streams. 
                    Use None for unconditional sampling (Note: in this case nSamples must not be None)
        nSamples    Number of samples to generate. Can be None if defined through the shape of the input data batch.

        Returns:

        A sample tensor or a list of sample tensors if this model has multiple input streams.
        """

        feed_dict={}
        if isinstance(inputs,DataIn):
            assert(self.numInputs==1)
            #If the caller passed in a single DataIn instance instead of a list of them, 
            #wrap the instance in a list, and remember to unwrap the return value
            inputs=[inputs]
        for i in range(len(self.inputs)):
            feed_dict.update(streamFeedDict(self.inputs[i],nSamples,feed=None if inputs is None else inputs[i]))
        sampleTensor=self.samples if sampledLayer is None else self.layerSamples[sampledLayer]
        if getLogp or sorted:
            if self.useBwdSampling and sampleInBwdPass:
                samples=self.sess.run(sampleTensor,feed_dict)
                for i in range(len(self.inputs)):
                    #Force the known/desired values of the samples to be correct.
                    #This allows more useful probability estimates for samples with incorrect values.
                    forcedSamples=samples[i]
                    if (inputs is not None) and (inputs[i] is not None) and (inputs[i].mask is not None):
                        thresholdedMask=np.clip(np.sign(inputs[i].mask),0,1)
                        #forcedSamples=inputs[i].data*inputs[i].mask+samples[i]*(1.0-inputs[i].mask)
                        forcedSamples = inputs[i].data * thresholdedMask + samples[i] * (1.0 - thresholdedMask)
                        newMask=np.maximum(thresholdedMask,inputs[i].mask)
                        feed_dict[self.inputs[i].tensor]=forcedSamples
                        feed_dict[self.inputs[i].mask]=newMask
                logProbabilities=self.sess.run(self.logp,feed_dict)
            else:
                samples,logProbabilities=self.sess.run([sampleTensor,self.logp],feed_dict)
            logProbabilities=logProbabilities[:,0]
        else:
            samples=self.sess.run(sampleTensor,feed_dict)
        if sorted:
            indices=np.argsort(-logProbabilities)
            for i in range(len(samples)):
                samples[i]=samples[i][indices]
            logProbabilities=logProbabilities[indices]
        if self.numInputs==1:
            samples=samples[0]
        if getLogp:
            return samples,logProbabilities
        else:
            return samples

    def getLogP(self,inputs,layerIdx=None):
        feed_dict={}
        if isinstance(inputs,DataIn):
            assert(self.numInputs==1)
            #If the caller passed in a single DataIn instance instead of a list of them,
            #wrap the instance in a list, and remember to unwrap the return value
            inputs=[inputs]
        for i in range(len(self.inputs)):
            feed_dict.update(streamFeedDict(self.inputs[i],feed=None if inputs is None else inputs[i]))
        return self.sess.run(self.logp if layerIdx is None else self.layers[layerIdx].logp,feed_dict)

#the following three methods are modified from from https://stackoverflow.com/questions/44047753/reconstructing-an-image-after-using-extract-image-patches     
def extract_patches(x,patchSize,stride):
    return tf.extract_image_patches(
        x,
        patchSize, #patch size
        stride, #stride
        (1, 1, 1, 1),
        padding="SAME"  
    )

#x is only for shape
def extract_patches_inverse_mean(y,x,patchSize,stride):
    _x = tf.zeros_like(x)
    _y = extract_patches(_x, patchSize,stride)
    grad = tf.gradients(_y, _x)[0]
    # Divide by grad, to "average" together the overlapping patches
    # otherwise they would simply sum up
    return tf.gradients(_y, _x, grad_ys=y)[0] / grad

def extract_patches_inverse_sum(y,x,patchSize,stride):
    _x = tf.zeros_like(x)
    _y = extract_patches(_x, patchSize,stride)
    return tf.gradients(_y, _x, grad_ys=y)[0]


class ExtractPatches(Layer):
    def __init__(self,patchSize,stride):
        Layer.__init__(self)
        self.patchSize=patchSize
        self.stride=stride
    def fwd(self,inputs,mode):
        self.inputs=inputs.copy()
        outputs=[]
        for stream in inputs:
            def process(t):
                if t is None:
                    return None
                assert(len(t.shape.as_list())==4)
                return extract_patches(t,self.patchSize,self.stride)
            #First, make a copy of the input stream
            stream=stream.copy()
            #In case weight is defined per sample, expand it to per pixel for the subsequent processing operations to work properly
            if len(stream.weights.shape.as_list())==2:
                stream.weights=tf.reshape(stream.weights,[-1,1,1,stream.weights.shape[1].value])
                #targetShape=stream.tensor.shape.as_list()
                #targetShape[0]=-1
                targetShape=tf.shape(stream.tensor)
                stream.weights=tf.broadcast_to(stream.weights,targetShape)
            #Process everything
            processStreamTensors(stream,process,broadcastToBatch=True)
            #Sum logQ because need a single logQ per patch
            if stream.logQ is not None:
                stream.logQ = tf.reduce_sum(stream.logQ, axis=-1,keepdims=True)
            #Average weights because need a single weight per patch
            if stream.weights is not None:
                stream.weights = tf.reduce_mean(stream.weights, axis=-1,keepdims=True)
            #Finalize
            outputs.append(stream)
        return outputs
    def bwd(self, data, mode):
        #Overlap-add patches to reconstruct the input to this block
        #Categorical pdf streams are first converted to log-domain, then converted back after the overlap-add
        #(i.e., the pdfs are combined multiplicatively)
        result=[]
        assert(len(data)==len(self.inputs))
        for streamIdx in range(len(data)):
            stream=data[streamIdx]
            tensor=stream.tensor
            #Now, overlap-average
            tensor=extract_patches_inverse_mean(tensor,self.inputs[streamIdx].tensor,self.patchSize,self.stride)
            if stream.type=="discrete":
                #ensure proper log-pdf shift
                tensor=streamLogSoftmax(tensor,stream.nCategories)
            result.append(DataStream(tensor,stream.type,mask=None,nCategories=stream.nCategories))
        return result        



class InterleavePatches(Layer):
    def __init__(self):
        Layer.__init__(self)
    def fwd(self,inputs,mode):
        self.inputs=inputs.copy()
        outputs=[]
        for stream in inputs:
            def process(t):
                if t is None:
                    return None
                assert(len(t.shape.as_list())==4)
                #if len(t.shape.as_list())==2:
                #    #Per-sample weights have this shape: For the training losses to work correctly, we must
                #    t=tf.reshap
                #    #For non-array tensors, i.e., sample weights
                #    return t
                return tf.reshape(t,[-1,t.shape[-1]])
            #First, make a copy of the input stream
            stream=stream.copy()
            #Process everything
            processStreamTensors(stream,process)
            #Finalize
            outputs.append(stream)
        return outputs
    def bwd(self, data, mode):
        if data is None:
            return None
        result=[]
        assert(len(data)==len(self.inputs))
        for streamIdx in range(len(data)):
            inputTensor=self.inputs[streamIdx].tensor
            stream=data[streamIdx]
            tensor=tf.reshape(stream.tensor,[-1,inputTensor.shape[1].value,inputTensor.shape[2].value,inputTensor.shape[3].value])            
            result.append(DataStream(tensor,stream.type,None,stream.nCategories))
        return result

class DeinterleavePatches(Layer):
    def __init__(self,interleavePatches):
        Layer.__init__(self)
        self.interleavePatches=interleavePatches
    def fwd(self,inputs,mode):
        self.inputs=inputs.copy()
        outputs=[]
        shape1=self.interleavePatches.inputs[0].tensor.shape[1].value
        shape2=self.interleavePatches.inputs[0].tensor.shape[2].value

        for stream in inputs:
            def process(t):
                if t is None:
                    return None
                assert(len(t.shape.as_list())==2)
                #if len(t.shape.as_list())<4:
                #    return t
                return tf.reshape(t,[-1,shape1,shape2,t.shape[1].value])
            #First, make a copy of the input stream
            stream=stream.copy()
            #Process everything
            processStreamTensors(stream,process)
            #Weights are per-sample, not per patch => reduce
            if stream.weights is not None:
                stream.weights=tf.reduce_mean(stream.weights,axis=[1,2])
            #Finalize
            outputs.append(stream)
        return outputs
    def bwd(self, data, mode):
        if data is None:
            return None
        result=[]
        for stream in data:
            tensor=tf.reshape(stream.tensor,[-1,stream.tensor.shape[3].value])            
            #mask=stream.mask
            #if mask is not None:
            #    mask=tf.reshape(mask,[-1,mask.shape[3].value*self.shape1*self.shape2])
            result.append(DataStream(tensor,stream.type,None,stream.nCategories))
        return result

#
# class Reshape(Layer):
#     def __init__(self,shape):
#         Layer.__init__(self)
#         self.shape=shape
#     def fwd(self,inputs,mode):
#         self.inputs=inputs.copy()
#         outputs=[]
#         for stream in inputs:
#             t=tf.reshape(stream.tensor,self.shape)
#             mask=tf.reshape(stream.mask,self.shape)
#             outputs.append(DataStream(t,stream.type,mask,stream.nCategories))
#         return outputs
#     def bwd(self, data, mode):
#         if data is None:
#             return None
#         result=[]
#         assert(len(data)==len(self.inputs))
#         for streamIdx in range(len(data)):
#             stream=data[streamIdx]
#             tensor=stream.tensor
#             origShape=tf.shape(self.inputs[streamIdx].tensor)
#             tensor=tf.reshape(tensor,origShape)
#             result.append(DataStream(tensor,stream.type,None,stream.nCategories))
#         return result

class Flatten(Layer):
    def __init__(self):
        Layer.__init__(self)
    def fwd(self,inputs,mode):
        self.inputs=inputs.copy()
        outputs=[]
        for stream in inputs:
            def process(t):
                if t is None:
                    return None
                if len(t.shape.as_list())==2:
                    return t
                assert(len(t.shape.as_list())==4)
                return tf.reshape(t,[-1,t.shape[1].value*t.shape[2].value*t.shape[3].value])
            #First, make a copy of the input stream
            stream=stream.copy()
            #Process everything
            processStreamTensors(stream,process)
            #Reduce weights and logQ to one per sample
            if stream.weights is not None:
                stream.weights=tf.reduce_mean(stream.weights,axis=-1,keepdims=True)
            if stream.logQ is not None:
                stream.logQ=tf.reduce_sum(stream.logQ,axis=-1,keepdims=True)
            #Finalize
            outputs.append(stream)
        return outputs
    def bwd(self, data, mode):
        if data is None:
            return None
        result=[]
        assert(len(data)==len(self.inputs))
        for streamIdx in range(len(data)):
            stream=data[streamIdx]
            tensor=stream.tensor
            w=self.inputs[streamIdx].tensor.shape[1].value
            h=self.inputs[streamIdx].tensor.shape[2].value
            tensor=tf.reshape(tensor,[-1,w,h,tensor.shape[1].value//(w*h)])
            result.append(DataStream(tensor,stream.type,None,stream.nCategories))    
        return result

#Helper class for discarding streams during forward pass and adding them back on backward pass.
#To propagate uncertainty and reduce variance, we set the masks of remaining streams to zero if all the discarded streams 
#were unknown (i.e., had zero masks). It is assumed that the discarded streams are the original inputs to a DRMM block, i.e.,
#if the discarded streams are unknown, the outputs should also be unknown. The output latent variable distribution already
#captures that uncertainty, but is inaccurate with small batch sizes.
class DiscardResiduals(Layer):
    def __init__(self,nDiscarded:int,blockLastLayer=True):
        Layer.__init__(self)
        self.nDiscarded=nDiscarded
        self.blockLastLayer=blockLastLayer
    def fwd(self,inputs,mode):
        self.inputs=inputs.copy()
        outputs=[]
        meanMask=0.0 
        for i in range(0,self.nDiscarded):
            meanMask+=tf.reduce_mean(inputs[i].mask,axis=-1,keepdims=True)
        #meanMask/=tf.reduce_max(meanMask)+epsilon
        meanMask=tf.nn.relu(tf.sign(meanMask))
        #Loop over kept streams, copy to output and multiply output masks with the means
        for i in range(self.nDiscarded,len(inputs)):
            outputs.append(inputs[i].copy())
            outputs[-1].mask*=meanMask
            #discard the logQ, as the next block will model the data independently
            #outputs[-1].logQ=None
        return outputs
    def bwd(self, data, mode):
        #If this is the last layer of a block and this is a training pass, we route the fwd pass outputs backwards after batch-averaging.
        #This is because of the staged training: The bwd bias correction training doesn't depend on the later stages.
        if self.blockLastLayer and mode=="training":
            return stopStreamGradients(multiStreamBatchAverage(self.inputs),1.0)
        if data is None:
            return None
        result=[]
        #Synthesize the discarded streams by assuming they are residuals with no information
        for i in range(self.nDiscarded):
            tensor=tf.zeros_like(self.inputs[i].tensor)
            result.append(DataStream(tensor,self.inputs[i].type,None,self.inputs[i].nCategories))
        #Pass the bwd input streams forward as is
        for stream in data:
            result.append(stream)
        return result


#Helper class for reshaping streams of data sequence tensors of shape [nBatch,sequenceLength,nVars] to [nBatch,sequenceLength,1,nVars],
#so that they can be modeled similar to images and other 2D data
class Reshape1Dto2D(Layer):
    def __init__(self):
        Layer.__init__(self)
    def fwd(self,inputs,mode):
        outputs=[]
        for stream in inputs:
            def process(t):
                if t is None:
                    return None
                if len(t.shape.as_list())<3:
                    return t
                return tf.reshape(t,[-1,t.shape[1].value,1,t.shape[2].value])
            #First, make a copy of the input stream
            stream=stream.copy()
            #process data
            stream.tensor=process(stream.tensor)
            #process known variables mask
            stream.mask=process(stream.mask)
            #process the Gaussian prior
            if stream.gaussianPrior is not None:
                stream.gaussianPrior.mean=process(stream.gaussianPrior.mean)            
                stream.gaussianPrior.sd=process(stream.gaussianPrior.sd)            
            #process box constraints
            if stream.boxConstraint is not None:
                stream.boxConstraint.minValues=process(stream.boxConstraint.minValues)
                stream.boxConstraint.maxValues=process(stream.boxConstraint.maxValues)
                stream.boxConstraint.minValueWeights=process(stream.boxConstraint.minValueWeights)
                stream.boxConstraint.maxValueWeights=process(stream.boxConstraint.maxValueWeights)
            #process linear inequality constraints
            for c in stream.ieqConstraints:
                c.a=process(c.a)
            #Finalize
            outputs.append(stream)
        return outputs
    def bwd(self, data, mode):
        if data is None:
            return None
        outputs=[]
        for stream in data:
            outputs.append(stream.copy())
            t=outputs[-1].tensor
            outputs[-1].tensor=tf.reshape(t,[-1,t.shape[1].value,t.shape[3].value])
            if outputs[-1].mask is not None:
                t=outputs[-1].mask
                outputs[-1].mask=tf.reshape(t,[-1,t.shape[1].value,t.shape[3].value]) 
        return outputs


class DRMMBlock2D(LayerStack):
    def __init__(self,width:int,height:int,nComponentsPerLayer: int, nLayers:int, kernelSize=[1,3,3,1], stride=[1,2,2,1], nDiscardedInputs=0,nNonLatentInputs=0,bwdSamplingTemperature=1.0,precisionRho=0,useMaximumLikelihoodLoss=False,gradStopStrength=1.0,trainingPhase=None):
        LayerStack.__init__(self)
        extractPatches=ExtractPatches(patchSize=kernelSize,stride=stride)
        self.add(extractPatches)
        interleavePatches=InterleavePatches()
        self.add(interleavePatches)
        for i in range(nLayers):
            self.add(RMM(nComponentsPerLayer,
                         nNonLatentInputs=nNonLatentInputs,
                         allowResample=False,
                         bwdSamplingTemperature=bwdSamplingTemperature,
                         precisionRho=precisionRho,
                         inputWidth=width,
                         inputHeight=height,
                         useMaximumLikelihoodLoss=useMaximumLikelihoodLoss,
                         gradStopStrength=gradStopStrength,
                         trainingPhase=trainingPhase,
                         isLastLayer=i==nLayers-1))
        self.lastRMM=self.layers[-1]
        self.add(DeinterleavePatches(interleavePatches))
        if nDiscardedInputs>0:
            self.add(DiscardResiduals(nDiscardedInputs))
 


#blockDefs is a list of dicts
class DRMMBlockHierarchy(LayerStack):
    def __init__(self,sess,inputs,blockDefs,lastBlockClasses,lastBlockLayers,useStagedTraining=True,initialLearningRate=0.001,finalEStepPrecision=1.0,train=True,endToEnd=True):
        LayerStack.__init__(self,sess)
        self.useStagedTraining=useStagedTraining
        self.sess=sess
        if type(inputs) is DataStream:
            inputs=[inputs]  #everything below assumes that we have a list of DataStream instances => convert to one
        self.inputs=inputs.copy()
        nStages=len(blockDefs)+1 if useStagedTraining else 1
        self.nStages=nStages

        #a placeholder for sampling temperature, will be used in sample() and initializing the sampling operations
        self.temperature=tf.placeholder_with_default(tf.constant(1.0,dtype=tf.float32),shape=[])

        #if input data is 1D sequences, reshape to 2D to allow reusing 2D array processing code for 1D sequences
        sequential=False
        if len(inputs[0].tensor.shape)==3:
            print("DRMMBlockHierarchy: Sequential data detected, reshaping to images")
            self.add(Reshape1Dto2D())
            sequential=True

        #Determine input width & height and check that all inputs have same width and height
        width=inputs[0].tensor.shape[1].value
        height=1 if sequential else inputs[0].tensor.shape[2].value 
        for stream in inputs:
            assert(inputs[0].tensor.shape[1].value==stream.tensor.shape[1].value and inputs[0].tensor.shape[2].value==stream.tensor.shape[2].value)


        #learning rate and E-step precision schedule
        useMaximumLikelihoodLoss=False #endToEnd
        self.trainingPhase=tf.Variable(dtype=tf.float32,initial_value=1.0,trainable=False)
        thresholdedPhase = tf.nn.relu(tf.sign(self.trainingPhase - 1.0 / 3.0))
        thresholdedPhase2 = tf.nn.relu(tf.sign(self.trainingPhase - 2.0 / 3.0))
        #decayingPhase=tf.clip_by_value(tf.square(1.0-self.trainingPhase),0,1)
        #decayingPhase = 1.0 - self.trainingPhase
        decayingPhase = tf.clip_by_value(1.0 - 3.0 * (self.trainingPhase - 2.0 / 3.0), 0.0,1.0)  # linearly decay from 1.0 to zero in the 3rd phase
        decayingPhase *= tf.clip_by_value(1.0 - thresholdedPhase2, 0.1, 1.0) #initially 1, then drop to 0.1 at third phase and decrease linearly to zero
        self.learningRate=initialLearningRate*decayingPhase
        #thresholdedPhase=tf.nn.relu(tf.sign(self.trainingPhase-1.0/2.0))
        #thresholdedPhase2=tf.nn.relu(tf.sign(self.trainingPhase-3.0/4.0))
        gradStopStrength=1.0-thresholdedPhase2 if endToEnd else 1.0
        self.precisionRho=finalEStepPrecision*thresholdedPhase

        self.stagePhases=tf.Variable(dtype=tf.float32,initial_value=np.ones(nStages),trainable=False)
        self.stageRhos=finalEStepPrecision*tf.nn.relu(tf.sign(self.stagePhases-1.0/3.0))

        #Add the DRMM blocks for pixel patches 
        stageLastLayers=[]
        nInputs=len(inputs)
        nResolutionLevels=len(blockDefs)
        resoIdx=0
        stageIdx=0
        self.blocks=[]
        for b in blockDefs:
            kernelSize=[1,b["kernelSize"],1,1] if sequential else [1,b["kernelSize"][0],b["kernelSize"][1],1]
            stride=[1,b["stride"],1,1] if sequential else [1,b["stride"][0],b["stride"][1],1]
            width=int(ceil(width/stride[1]))
            height=int(ceil(height/stride[2]))
            self.add(DRMMBlock2D(width=width,height=height,
                           nComponentsPerLayer=b["nClasses"],
                           nLayers=b["nLayers"],
                           kernelSize=kernelSize,
                           stride=stride,                           
                           nNonLatentInputs=nInputs,nDiscardedInputs=nInputs,
                           bwdSamplingTemperature=self.temperature*max([epsilon,b.get("temperature",0.0)]),
                           precisionRho=self.stageRhos[stageIdx] if train else finalEStepPrecision,
                           useMaximumLikelihoodLoss=useMaximumLikelihoodLoss,
                           gradStopStrength=gradStopStrength, 
                           trainingPhase=self.trainingPhase))      
            self.blocks.append(self.layers[-1])
            nInputs=b["nLayers"]
            if useStagedTraining:
                stageLastLayers.append(len(self.layers))
                stageIdx+=1
            resoIdx+=1

        #Flatten all streams
        self.add(Flatten())

        #Add final layer(s) to model the flattened data
        for i in range(lastBlockLayers):
            self.add(RMM(lastBlockClasses,
                         nNonLatentInputs=nInputs,
                         precisionRho=self.stageRhos[-1] if train else finalEStepPrecision,
                         bwdSamplingTemperature=self.temperature,
                         useMaximumLikelihoodLoss=useMaximumLikelihoodLoss,
                         gradStopStrength=gradStopStrength,
                         trainingPhase=self.trainingPhase,
                         isLastLayer=i==lastBlockLayers-1))

        #if useStagedTraining:
        stageLastLayers.append(len(self.layers))
            
        #Construct graphs 
        self.build(inputs,bwdSampling=True)

        #Ops for correcting the likelihood estimates computed in the build() method above, which only considers the very last layer.
        #Each DRMMBlock2D discards its non-latent inputs, assuming that the later layers do not need to model them. 
        #Thus, we need to incorporate the likelihood conditional on the discarded streams here.
        for block in self.blocks:
            patchLogProbabilities=deinterleaveArray(block.lastRMM.nonLatentLogP,block.lastRMM.inputWidth,block.lastRMM.inputHeight)
            self.logp+=tf.reshape(tf.reduce_sum(patchLogProbabilities,axis=[1,2,3]),[-1,1])
        self.p=tf.exp(self.logp)

        #Losses. note: staged training losses initialized by default in LayerStack.Build()
        # if not useStagedTraining:
        #     self.stageLosses=[]
        #     self.stageVariables=[]
        #     #Final stage with all losses
        #     totalLoss=self.loss
        #     if useBwdCorrection:
        #         totalLoss+=self.bwdLoss
        #     self.stageLosses.append(totalLoss)
        #     self.stageVariables.append(self.getVariables())
        # else:
        self.stageLosses=[]
        self.stageVariables=[]

        stageLoss=0.0
        stageVariables=[]
        bwdLoss=0.0
        bwdVariables=[]
        stageIdx=0
        #helpers
        def getBwdVariables(layer):
            result=[]
            if hasattr(layer, "bwdVariables"):
                result.append(layer.bwdVariables)
            if isinstance(layer,LayerStack):
                for l in layer.layers:
                    if hasattr(l,"bwdVariables"):
                        result.append(l.bwdVariables)
            return result
        def joinWithoutDuplicates(list1,list2):
            result=list1.copy()
            for i in list2:
                if not i in result:
                    result.append(i)
            return result
        #Every stage optimizes the sum of losses up to the stage's block,
        #affecting the block's variables + all bwd corrections before it
        for L in range(len(self.layers)):
            stageLoss+=self.fwdLosses[L]
            if sampleInBwdPass:
                bwdLoss+=self.bwdLosses[L]
                bwdVariables=bwdVariables+getBwdVariables(L)
            stageVariables=stageVariables+self.layers[L].getVariables()
            if L==stageLastLayers[stageIdx]-1:
                self.stageLosses.append(stageLoss+bwdLoss)
                self.stageVariables.append(joinWithoutDuplicates(stageVariables,bwdVariables))
                stageLoss=0.0
                stageVariables=[]
                stageIdx+=1

        #Create optimizer
        if train:
            self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learningRate)
            self.optimizeOps=[]
            for stageIdx in range(len(self.stageLosses)):
                #gradients, variables = zip(*self.optimizer.compute_gradients(self.stageLosses[stageIdx],var_list=self.stageVariables[stageIdx]))
                #gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
                #optimizeOp = self.optimizer.apply_gradients(zip(gradients, variables))
                #self.pretrainOps.append(optimizeOp)
                #self.optimizeOps.append(optimizeOp)
                self.optimizeOps.append(self.optimizer.minimize(self.stageLosses[stageIdx],var_list=self.stageVariables[stageIdx]))


    def train(self,phase,dataBatch):
        nStages=len(self.stageLosses)
        stage=int(np.clip(phase*nStages,0,nStages-1))
        stagePhase=(phase-stage*(1.0/nStages))*nStages
        self.trainingPhase.load(stagePhase,self.sess)
        stagePhases=np.ones(nStages)
        stagePhases[stage]=stagePhase
        self.stagePhases.load(stagePhases,self.sess)
        feed_dict={}
        if not isinstance(dataBatch,list):
            dataBatch=[dataBatch]
        for i in range(len(self.inputs)):
            feed_dict[self.inputs[i].tensor]=dataBatch[i]
            feed_dict[self.inputs[i].mask]=np.ones_like(dataBatch[i])
            feed_dict[self.inputs[i].weights]=np.ones([dataBatch[i].shape[0],1])

        currLoss,currLearningRate,currPrecision,temp=self.sess.run([self.stageLosses[stage],self.learningRate,self.precisionRho,self.optimizeOps[stage]],feed_dict)
        return {"loss":currLoss,"lr":currLearningRate,"rho":currPrecision,"stage":stage+1,"nStages":nStages}
    def sample(self,nSamples=None,inputs=None,temperature=1.0,getLogp=False,sorted=True):
        """
        Generate a batch of samples.

        Parameters:
        inputs      DataIn instance or a list of DataIn instances if this model has multiple input streams. 
                    Use None for unconditional sampling (Note: in this case nSamples must not be None)
        nSamples    Number of samples to generate. Can be None if defined through the shape of the input data batch.

        Returns:

        A sample tensor or a list of sample tensors if this model has multiple input streams.
        """

        feed_dict={}
        if isinstance(inputs,DataIn):
            assert(self.numInputs==1)
            #If the caller passed in a single DataIn instance instead of a list of them, 
            #wrap the instance in a list, and remember to unwrap the return value
            inputs=[inputs]
        for i in range(len(self.inputs)):
            feed_dict.update(streamFeedDict(self.inputs[i],nSamples,feed=None if inputs is None else inputs[i]))
        feed_dict[self.temperature]=temperature
        samples=self.sess.run(self.samples,feed_dict)
        if getLogp or sorted:
            if sampleInBwdPass:
                #Because of backward sampling, we must query the probabilities in a new batch from the generated samples
                #In doing this, we treat the samples as known inputs
                samples=self.sess.run(self.samples,feed_dict)
                for i in range(len(self.inputs)):
                    #Force the known/desired values of the samples to be correct.
                    #This allows more useful probability estimates for samples with incorrect values.
                    if (inputs is not None) and (inputs[i] is not None) and (inputs[i].mask is not None):
                        #forcedSamples=inputs[i].data*inputs[i].mask+samples[i]*(1.0-inputs[i].mask)
                        thresholdedMask = np.clip(np.sign(inputs[i].mask), 0, 1)
                        forcedSamples = inputs[i].data * thresholdedMask + samples[i] * (1.0 - thresholdedMask)
                        newMask = np.maximum(thresholdedMask, inputs[i].mask)
                        feed_dict[self.inputs[i].tensor]=forcedSamples
                        feed_dict[self.inputs[i].mask]=np.ones_like(samples[i])
                logProbabilities=self.sess.run(self.logp,feed_dict)

            else:
                samples,logProbabilities = self.sess.run([self.samples,self.logp], feed_dict)
            logProbabilities=logProbabilities[:,0]
        if sorted:
            indices=np.argsort(-logProbabilities)
            for i in range(len(samples)):
                samples[i]=samples[i][indices]
        if self.numInputs==1:
            samples=samples[0]
        if getLogp:
            return samples,logProbabilities
        else:
            return samples

