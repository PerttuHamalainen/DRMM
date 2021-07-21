import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from collections import Counter
from matplotlib import rc
import argparse

#LaTeX font stuff, commented out
#rc('text', usetex=True)
#rc('font',**{'family':'serif','serif':['Times']})


parser = argparse.ArgumentParser()
parser.add_argument('--windowless', default=False,action='store_true')
args = parser.parse_args()

#data=pd.read_csv("Results/benchmark_precrecall_resampling=False.csv",delimiter=',')
data=pd.read_csv("Results/benchmark_precrecall.csv",delimiter=',')
print(data["nParameters"])
datasets=["Full body IK, 30D","Motion Capture, 1800D"]
metrics=["f1","logp"]
metricLabels=["F1 score","Log-likelihood"]
layerAmounts=[1,2,3,4]
nCurvePoints=5
plotHScale=3.5
plotVScale=2.5*1.1
fontSize=14
legendFontSize=10
singleRow=True

colors=["gray","blue","red","green"]

if singleRow:
    pp.figure(1,figsize=[len(datasets)*2*plotHScale,plotVScale])
else:
    pp.figure(1,figsize=[len(datasets)*plotHScale,2*plotVScale])

nPlots=len(datasets)*2
for metricIdx in range(len(metrics)):
    rowIdx = 0
    metric=metrics[metricIdx]
    for datasetIdx in range(len(datasets)):
        dataset=datasets[datasetIdx]
        for curveIdx in range(len(layerAmounts)):
            nLayers=layerAmounts[curveIdx]
            nParams=[]
            y=[]
            nComponents=[]
            for i in range(nCurvePoints):
                nParams.append(data["nParameters"][rowIdx])
                y.append(data[metric][rowIdx])
                nComponents.append(nLayers*data["nComponentsPerLayer"][rowIdx])
                rowIdx+=1

            if singleRow:
                pp.subplot(1,nPlots,metricIdx*len(datasets)+datasetIdx+1)
            else:
                pp.subplot(2, nPlots//2, metricIdx * len(datasets) + datasetIdx + 1)
            pp.plot(nParams, y,label="{} layers".format(nLayers), color=colors[curveIdx])
            if singleRow:
                pp.title("Dataset: {}".format(dataset), fontSize=fontSize)
                pp.xlabel("Model parameters",fontSize=fontSize)
                pp.ylabel(metricLabels[metricIdx], fontSize=fontSize)
            else:
                if datasetIdx==0:
                    pp.ylabel(metricLabels[metricIdx], fontSize=fontSize)
                if metricIdx==len(metrics)-1:
                    pp.xlabel("Model parameters", fontSize=fontSize)
                if metricIdx==0:
                    pp.title("Dataset: {}".format(dataset), fontSize=fontSize)

            pp.legend(fontsize=legendFontSize, framealpha=0.9, loc='lower right')
            pp.tight_layout(pad=0.2)

if singleRow:
    pp.savefig("images/benchmark_singlerow.png",dpi=200)
else:
    pp.savefig("images/benchmark.png",dpi=200)

#pp.savefig("images/benchmark.pdf",dpi=200)
import os
if os.path.exists("PNGCrop.py"):
    import PNGCrop

    if singleRow:
        PNGCrop.crop("images/benchmark_singlerow.png")
    else:
        PNGCrop.crop("images/benchmark.png")
if not args.windowless:
    pp.show()
