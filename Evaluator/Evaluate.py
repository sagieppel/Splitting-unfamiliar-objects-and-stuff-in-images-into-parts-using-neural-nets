# Evaluate net performance of the evalautor
# The evaluation data need to be prepared  by the pointer net (see script GenerateTrainingDataForEvaluator.py in pointer net for generation)
#...............................Imports..................................................................
import os
import torch
import numpy as np
import ReaderParts
import NetModel as NET_FCN # The net Class
import scipy.misc as misc
##################################Input paramaters#########################################################################################
AnnDir="../TrainingDataForEvaluator//Ann/"
ImageDir="../TrainingDataForEvaluator//Img/"


Trained_model_path = "logs/600000.torch"
##################################Input folders#########################################################################################


#########################Params unused######################################################################33
NumClasses=205
MaxBatchSize=7 # Max images in batch
MinSize=250 # Min image Height/Width
MaxSize=1000# Max image Height/Width
MaxPixels=340000*4# Max pixel in batch can have (to keep oom out of memory problems) if the image larger it will be resized.
MinPrecision=0.0
#=========================Load Paramters====================================================================================================================

#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=NET_FCN.Net() # Create net and load pretrained encoder path
Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.cuda()
Net.eval()
Net.half()

#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader = ReaderParts.Reader(ImageDir=ImageDir,MaskDir=AnnDir, MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,TrainingMode=False,InverseSelection=False)

#.............. Evaluating....................................................................
print("Start Evaluating")
ErrIOU={}
Nclass={}
for itr in range(0,5000): # Mainevaluation loop
    print(itr)
    Img,Mask, ROIMask,GTIOU,Cat,Finished = Reader.LoadSingle()

    if Finished: break
#--------------------------------------------------------------------------------------------------
    # Img[0,:,:,0] *=1 - Mask[0,:,:]
    # Img[0, :, :, 1] *= 1 - ROIMask[0, :, :]
    # print("IOU="+str(GTIOU))
    # print(Cat)
    # misc.imshow(Img[0])
#----------------------------------------------------------------------
    PredIOU = Net.forward(Img, Segment=Mask,ROI=ROIMask, TrainMode=False)
    PredIOU=float(PredIOU.data.cpu().numpy())
    if not Cat in  ErrIOU:
        ErrIOU[Cat]=0
        Nclass[Cat]=0
    ErrIOU[Cat]+=abs(PredIOU-GTIOU)
    Nclass[Cat]+=1


#===================================================================================
    NumClasses=0
    SumErrIOU=0
    for cl in Nclass:
        SumErrIOU+=ErrIOU[cl]/Nclass[cl]
        NumClasses+=1
    print("Number of Classes="+str(NumClasses)+"         Average error per class=\t"+str(SumErrIOU/NumClasses))



