'''

This Python script should train DRMM models and generate all the results and software-generated figures in the paper.
The figures will be saved in the "images" folder in .png format.

''' 

import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--quicktest', default=False,action='store_true')
args = parser.parse_args()

#Some parameters depending on whether we want to actually run everything or just quickly test that the code runs
if args.quicktest:
    # iteration counts for different tests.
    n2dIter = 1000
    nIKIter = 1000
    nMocapIter = 1000
    nPrecRecallIter = 1000

    # Precision and recall estimation samples
    nPrecRecallEval = 1000

    # How much IK data to generate.
    nIKData = 10000

    # Whether to show results after each test and wait for the user to close the window
    windowless = True

else:
    #iteration counts for different tests.
    n2dIter=50000
    nIKIter=500000
    nMocapIter=500000
    nPrecRecallIter=50000

    # Precision and recall estimation samples
    nPrecRecallEval = 20000

    #How much IK data to generate.
    nIKData=1000000

    # Whether to show results after each test and wait for the user to close the window
    windowless=True


#Create directories
if not os.path.exists('Results'):
    os.makedirs('Results')
if not os.path.exists('trainedmodels'):
    os.makedirs('trainedmodels')
if not os.path.exists('IKTest'):
    os.makedirs('IKTest')
if not os.path.exists('imagetemp'):
    os.makedirs('imagetemp')
if not os.path.exists('images'):
    os.makedirs('images')

#Delete old images
#for f in glob.glob("images/*.png"):
#    os.remove(f)

#2d data with both samples and likelihood plots
if os.system("python Plot_ToyData.py --nIter={} {}".format(n2dIter,"--windowless" if windowless else "")):
    exit()

#A visualization of residuals, omitted in the current version
#if os.system("python Plot_Residuals.py --nIter=20000 --windowless"):
#    exit()

#Curriculum visualization
if os.system("python Plot_curriculum.py --nIter={} {}".format(n2dIter,"--windowless" if windowless else "")):
    exit()

#Inverse Kinematics
if os.system("python Plot_IK.py --train --nData={} --nIter={} {}".format(nIKData,nIKIter,"--windowless" if windowless else "")):
    exit()

#Animation synthesis using the motion capture data. By default, we don't render the videos, as it requires ffmpeg utilities.
#The script will leave the individual frames as .png images in the imagetemp folder
if os.system("python Mocaptest_conditional.py --train --nIter={} --nVideoTakes=1 --videoTakeLength=300 {}".format(nMocapIter,"--windowless" if windowless else "")):
    exit()

#Compute F1 scores and log-likelihoods to a .csv file
if os.path.isfile('Results/benchmark_precrecall.csv'):
    os.remove('Results/benchmark_precrecall.csv')
for datasetIdx in range(2):
    for modelIdx in range(4):
        if os.system(
                "python PrecisionRecallTest.py  --datasetIdx={} --modelIdx={} --nIter={} --nEval={}".format(datasetIdx, modelIdx,nPrecRecallIter,nPrecRecallEval)):
            exit()

#Plot the F1 scores
if os.system("python Plot_f1.py {}".format("--windowless" if windowless else "")):
    exit()

