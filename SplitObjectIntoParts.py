# Main function for splitting object into parts


import os
import torch
import numpy as np
import scipy.misc as misc


import cv2
##################################Make sre matchjng GT and pred segments will have same index######################################################################
def MatchIndexses(GT,Prd):#Make sre matchjng GT and pred segments will have same index
    for g in np.unique(GT):
        if g==0: continue
        Gmask=(GT==g)
        for p in np.unique(Prd):
            if p == 0: continue
            Pmask = (Prd == p)
            IOU=(Pmask*Gmask).sum()/(Pmask.sum()+Gmask.sum()-(Pmask*Gmask).sum())
            if IOU>=0.5:
                GT[GT==p]=g
                GT[Gmask] = p
    return GT


###########################################################################################################################################################3
##############################Annotate object region into parts main annotation function#########################################################################################################################################################
def SplitObjectToParts(Img,ObjectMask,Generator,Evaluator,MinIOUThresh=-1,MaxOverlap=0.3): # given an image an object mask split region object parts
            H,W=ObjectMask.shape
            InsList = np.zeros([0, H,W]) #List of instances predicted by the net
            InstRank = [] # IOU score for the predicted instances
            NumPoints= int(340000 * 12/(H*W)) # Num instace points to guess per cycle
            OccupyMask = np.zeros([H, W], int)  # Region that already been segmented
            if NumPoints<=1: NumPoints=1


# .........................Generate input for the net................................................................

            ROI = np.ones([NumPoints, H, W], dtype=float)
            ImgList= np.ones([NumPoints, H, W,3], dtype=float)
            for i in range(NumPoints): # generate pointer mask
                    ImgList[i]=Img
                    ROI[i]=ObjectMask.copy()
##==============Collect segments==========================================================================
            while(not ((InsList.shape[0]>80 and (OccupyMask.sum()/ObjectMask.sum())>0.95) or InsList.shape[0]>200)):
                PointerMask = np.zeros([NumPoints, H, W], dtype=float)
                for i in range(NumPoints):  # generate pointer maskw
                    while (True):
                        px = np.random.randint(W)
                        py = np.random.randint(H)
                        if (ObjectMask[py, px]) == 1: break
                    PointerMask[i, py, px] = 1
                #*******************************************************************************************************
                #******************************************************************************************************
                # print("From readed")
                # for f in range(PointerMask.shape[0]):
                #     ImgList[f, :, :, 1] *= ObjectMask.astype(np.uint8)
                #     misc.imshow(ImgList[f])
                #     misc.imshow((ROI[f] * 2- PointerMask[f] * 3).astype(np.uint8)*40)
                # #******************************************************************************************************
                #******************************************************************************************************
    #-----------------------Run Generator and evaluator  Net------------------------------------------------
                with torch.autograd.no_grad():
                    Prob, Lb = Generator.forward(Images=ImgList, Pointer=PointerMask, ROI=ROI, TrainMode=False)
                   # Prob, Lb, PredIOU, Predclasslist = MatNet.forward(Images=ImgList, Pointer=PointerMask,ROI=ROI,TrainMode=False,UseGPU=UseGPU, FreezeBatchNorm_EvalON=FreezeBatchNorm_EvalON)
                    Masks = Lb.data.cpu().numpy().astype(float)
                    PredIOU = Evaluator.forward(ImgList, Segment=Masks, ROI=ROI, TrainMode=False)
                    IOU = PredIOU.data.cpu().numpy().astype(float)
    # *******************************Add segments and scores to list************************************************************************

                    for f in range(NumPoints):
                            InsList = np.concatenate([InsList,np.expand_dims(Masks[f],0)],axis=0)
                            if NumPoints==1: IOU=[IOU]
                            InstRank.append(IOU[f])
                            OccupyMask[Masks[f]>0]=1 #
                          #  print(InsList.shape[0])

                #********************************************************************************************
                # print("From readed")
                # for f in range(PointerMask.shape[0]):
                #     print(IOU[f])
                #     I=ImgList[f]
                #     I[ :, :, 1] *= 1 - ObjectMask.astype(np.uint8)
                #     I[:, :, 2] *= 1 - Masks[f].astype(np.uint8)
                #     misc.imshow(I)
                #     misc.imshow((ROI[f] * 2 +  PointerMask[f] * 7).astype(np.uint8) * 40)
                #     misc.imshow((ROI[f] * 2+Masks[f]+ PointerMask[f] * 7).astype(np.uint8)*40)
                # #******************************************************************************************************
    #=========================================================================================================================
    #==============================Create final annotation map by combining generated segments in the order of their score===============================================================================================================================================================
            OccupyMask = np.zeros([H, W], int)  # Region that already been segmented
            InstMap = np.zeros([H, W], int)  # Map of all approved instances
            NumInst=1
            while np.max(InstRank)>MinIOUThresh:
                    ind=np.argmax(InstRank)
                    Seg=InsList[ind]

                    #******************************************
                    #print(InstRank[ind])
                    #misc.imshow((Seg * 6 ).astype(np.uint8) * 40)
                    #misc.imshow((ROI[f] * 1 + Seg*6+OccupyMask*1).astype(np.uint8) * 40)
                    #*******************************************************
                    InstRank[ind]=MinIOUThresh-1

                    Intr=(Seg*OccupyMask)
                    if (Intr.sum()/Seg.sum())>MaxOverlap: continue
                    Seg[Intr>0]=0
                    #*******************************************************
                    #misc.imshow((ROI[f] * 1 + Seg * 6 ).astype(np.uint8) * 40)
                    # *******************************************************
                    InstMap[Seg>0]=NumInst
                    OccupyMask[Seg>0]=1
                    NumInst+=1



            return InstMap#, OccupyMask,NInst,InsList,InstRank,
