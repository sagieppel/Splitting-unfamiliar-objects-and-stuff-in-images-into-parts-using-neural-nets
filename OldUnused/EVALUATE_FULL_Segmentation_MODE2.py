#..............Imports..................................................................
import os
import torch
import PointerSegmentation.ReaderADE_Parts as Data_Reader
import Evaluation.NetModel as EvaluatorNet
import numpy as np
import PointerSegmentation.FCN_NetModel as GeneratorNet # The net Class
import cv2
import scipy.misc as misc
#modelDir="logs/"


#.................................Main Input parametrs...........................................................................................
MinIOUThresh=-10
MaxOverlap=0.5
Generator_model_path = "/PointerSegmentation/logs/1200000.torch"
Evaluator_model_path = "logs/600000.torch"
MaskDir=r"/media/sagi/2T/Data_zoo/ADE20K_2016_07_26/LabelConvertedValidation2/"
ImageDir=r"/media/sagi/2T/Data_zoo/ADE20k_PARTS/ImageConvertedValidation//"
WriteAnn=True
if WriteAnn:
   OutDir=r"/media/sagi/2T/Data_zoo/ResultsGESParts_Mod2/"
   if not os.path.isdir(OutDir): os.mkdir(OutDir)



Generator_model_path = "PointerSegmentation/logs/1200000.torch"
Evaluator_model_path = "Evaluation/logs/600000.torch"
#IOUthresh=0.8
print("loading nets")
##################################Load Evaluator net#########################################################################################
Evaluator=EvaluatorNet.Net() # Create net and load pretrained encoder path
Evaluator.load_state_dict(torch.load(Evaluator_model_path))
Evaluator=Evaluator.cuda()
Evaluator.eval()
Evaluator.half()

##################################Load Generator net###############################################################################
#---------------------Create and Initiate net-----------------------------------------------------------------------------------
Generator=GeneratorNet.Net(NumClasses=2) # Create net and load pretrained
Generator=Generator.cuda()
Generator.load_state_dict(torch.load(Generator_model_path))
Generator.eval()
Generator.half()
########################################################################################################################################

#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader=Data_Reader.Reader(ImageDir,MaskDir,TrainingMode=False,InverseSelection=True)# MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,TrainingMode=False)

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
##################################Input paramaters#########################################################################################

###########################################################################################################################################################3
############################################################Main generator evaluator function#########################################################################################################################################################
def SplitObjectToParts(Img,ObjectMask): # Split  vessel region to materials instance and empyty region instances and classify material
            H,W=ObjectMask.shape
            InsList = np.zeros([0, H,W]) #List of instances
            InstRank = [] # IOU score for the instances
            NumPoints= int(340000 * 12/(H*W)) # Num instace points to guess per cycle
            OccupyMask = np.zeros([H, W], int)  # Region that already been segmented
            if NumPoints<=1: NumPoints=1
#===============Generate instance map========================================================================================

# .........................Generate input for the net................................................................

            ROI = np.ones([NumPoints, H, W], dtype=float)
            ImgList= np.ones([NumPoints, H, W,3], dtype=float)
            for i in range(NumPoints): # generate pointer mask
                    ImgList[i]=Img
                    ROI[i]=ObjectMask.copy()
##==============Collect segments==========================================================================
            while(not ((InsList.shape[0]>80 and (OccupyMask.sum()/ObjectMask.sum())>0.95) or InsList.shape[0]>200)):
                PointerMask = np.zeros([NumPoints, H, W], dtype=float)
                for i in range(NumPoints):  # generate pointer mask
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
    #-----------------------Run Generator Net------------------------------------------------
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
                            print(InsList.shape[0])

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
    #==============================Create final annotation map===============================================================================================================================================================
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

##############################################################################################################################################################################################################################################
#########################################################################################################################################################################


print("Start Evaluating ")
Siou={}
Sprec={}
Srecall={}
Sn={}


Con=True

#------------------------------------Evaluation loop---------------------------------------------------------------------------------
for i in range(4000):

        #Img, Mask, PointerMask, ROIMask, CatID, sy, sx = Reader.LoadSingle(ByClass=True)
        Img, Mask, PointerMask, ROIMask, FullAnnGT, fname, PartLevel, PartClass, Class = Reader.LoadSingle()
        if (Mask.sum()/ROIMask.sum())<0.02 or Mask.sum()<200: continue#***********************************************************
        print(str(i)+"File name")
        print(fname)
    #****************Resize if to big***********************************************************

        h=Mask.shape[1]
        w=Mask.shape[2]
    #    if not fname=="ADE_val_00000463.jpg": continue
        MaxPixels=3500000
        if h*w>MaxPixels:
            rt=np.sqrt(MaxPixels/(h*w))
            h=int(rt*h)
            w=int(rt*w)
            Img=np.expand_dims(cv2.resize(Img[0],(w,h),interpolation=cv2.INTER_LINEAR),axis=0)
            FullAnnGT = cv2.resize(FullAnnGT, (w, h), interpolation=cv2.INTER_NEAREST)
            Mask= np.expand_dims(cv2.resize(Mask[0], (w, h), interpolation=cv2.INTER_NEAREST),axis=0)
            ROIMask= np.expand_dims(cv2.resize(ROIMask[0], (w, h), interpolation=cv2.INTER_NEAREST),axis=0)

        #     Con=False
        # if Con: continue
    #***********************************************************************************************************************************************
        # for f in range(Img.shape[0]):
        #     Img[f, :, :, 0] *= 1-Mask[f]
        #     Img[f, :, :, 1] *= ROIMask[f]
        #     Img[f, :, :, 1] *= FullAnnGT
        #     misc.imshow((ROIMask[f] + Mask[f] * 2 + PointerMask[f] * 3).astype(np.uint8)*40)
        #     misc.imshow(Img[f])
        # print(ROIMask.shape)
    #***************************************Start loop tht segment the entire objects into parts  by part and pick the segment  best fitted to the target segment in term of IOU***********************************************************************************************************
        SegmentTop=np.zeros([Img.shape[1],Img.shape[2]],dtype=np.float32) # List of all the parts masks (Parts can overlap)
        PrecisionTop=0 # Precision of each part mask and the target mask
        IOUTop = 0 # IOU
        RecallTop = 0 # Object region tht reamin unsegmented


        SegmentationMap=SplitObjectToParts(Img[0], ROIMask[0])

        for ff in np.unique(SegmentationMap): # Main segmenting loop find all the parts of the object in the ROI region
            if ff==0: continue
            Pred=(SegmentationMap==ff)

            Inter=(Pred*Mask).sum()
            Gs=Mask.sum()
            Ps=Pred.sum()
            IOU=Inter/(Gs+Ps-Inter)
            Precision=Inter/(Ps+0.0001)
            Recall=Inter/(Gs+0.00001)
            if IOU>IOUTop:
                IOUTop=IOU
                PrecisionTop=Precision
                RecallTop=Recall
                SegmentTop=Pred





        IOU=IOUTop
        Precision=PrecisionTop
        Recall=RecallTop
        Pred=SegmentTop

    #********************Display segmentation map******************************************************************************
        if WriteAnn:
            Img[0] = Img[0][..., ::-1]
            ROIMask = np.expand_dims(ROIMask[0], axis=2)
            ROIMask=np.concatenate([ROIMask*220,ROIMask*1,ROIMask*220],axis=2)

            FullAnnGT=MatchIndexses(FullAnnGT,SegmentationMap)

            VizSegMapPr=np.expand_dims(SegmentationMap,axis=2)
            VizSegMapPr=np.concatenate([(VizSegMapPr*317)%255,(VizSegMapPr*17)%255,(VizSegMapPr*110)%255],axis=2).astype(np.uint8)
            FullAnnGT = np.expand_dims(FullAnnGT, axis=2)
            VizSegMapGT = np.concatenate([(FullAnnGT * 317) % 255, (FullAnnGT * 17) % 255, (FullAnnGT * 110) % 255],axis=2).astype(np.uint8)
           # misc.imshow(VizSegMap)
            Sep=np.ones([Img[0].shape[0],20,3],np.uint8)*255
            Overlay=np.concatenate([Img[0],Sep,ROIMask*0.7+Img[0]*0.3,Sep,VizSegMapGT * 0.7 + Img[0] * 0.3,Sep, VizSegMapPr*0.7+Img[0]*0.3],axis=1)
            Viz = np.concatenate([Img[0],Sep,ROIMask, Sep, VizSegMapGT , Sep, VizSegMapPr],axis=1)

            I1 = Img[0].copy()
            I1[:, :][ROIMask>0 ]=0
            Overlay2 = np.concatenate([Img[0], Sep, ROIMask+ I1*0.5, Sep, VizSegMapGT + I1*0.5 , Sep,VizSegMapPr + I1*0.5], axis=1)



            # misc.imshow(Overlay)
            # misc.imshow(Viz)
            cv2.imwrite(OutDir+"/"+str(i)+"_"+fname[:-4]+".png",Viz)
            cv2.imwrite(OutDir+"/"+str(i)+"_"+fname[:-4]+"Overlay.jpg", Overlay)
            cv2.imwrite(OutDir + "/" + str(i) + "_" + fname[:-4] + "Overlay2.png", Overlay2)

    #**********************************************************************************************************************************************************************
        if not Class in Siou:
            Siou[Class] = IOU
            Sprec[Class] = Precision
            Srecall[Class] = Recall
            Sn[Class] = 1

        Siou[Class] += IOU
        Sprec[Class] += Precision
        Srecall[Class] += Recall
        Sn[Class]+=1
    #****************************Visualize**************************************************************************************************************************
        # print("IOU="+str(IOU))
        # print("Precision=" + str(Precision))
        # print("Recall=" + str(Recall))
        #
        # Img[0, :, :, 0] *= 1-Mask[0]
        # Img[0, :, :, 1] *= 1-Pred[0]
        # misc.imshow(Img[0])

    #******************************************************************************************************************************************************

        SiouAr= np.array(list(Siou.values()))
        SprecAr= np.array(list(Sprec.values()))
        SrecallAr= np.array(list(Srecall.values()))
        SnAr= np.array(list(Sn.values()))

        #******************************************************************************************************************************************************
        Iou=(SiouAr/(SnAr+0.000001)).sum()/(SnAr>0).sum()
        Precision=(SprecAr/(SnAr+0.000001)).sum()/(SnAr>0).sum()
        Recall=(SrecallAr/(SnAr+ 0.000001)).sum()/(SnAr>0).sum()
        txt=Evaluator_model_path+"\n "+Generator_model_path+"Num Class=\t"+str((SnAr>0).sum())+"\tNum="+str(SnAr.sum())+"\tIOU="+str(Iou)+"\tPrecission="+str(Precision)+"\tRecall="+str(Recall)
        print(txt)
    # fl=open("logs/EvalResults.txt","a")
    # fl.write(txt)
    # fl.close()





# Results Unfamiliar logs//1200000.torchNum Class=	38	Num=4038	IOU=0.5047560520268203	Precission=0.6983280529896474	Recall=0.700490212312625



