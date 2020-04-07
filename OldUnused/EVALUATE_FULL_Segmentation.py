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
#.................................Main Input parametrs...........................................................................................
Generator_model_path = "/PointerSegmentation/logs/1200000.torch"
Evaluator_model_path = "logs/600000.torch"
MaskDir=r"/media/sagi/2T/Data_zoo/ADE20K_2016_07_26/LabelConvertedValidation2/"
ImageDir=r"/media/sagi/2T/Data_zoo/ADE20k_PARTS/ImageConvertedValidation//"
WriteAnn=True
if WriteAnn:
   OutDir=r"/media/sagi/2T/Data_zoo/ResultsGESParts/"
   if not os.path.isdir(OutDir): os.mkdir(OutDir)



Generator_model_path = "PointerSegmentation/logs/1200000.torch"
Evaluator_model_path = "Evaluation/logs/600000.torch"
NumSegCycles=80 # Number of segmentation cycles per object
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

###########################################################################################################################################################3
############################################################Main generator evaluator function#########################################################################################################################################################
def SplitObjectToParts(Img,ObjectMask): # Split  vessel region to materials instance and empyty region instances and classify material
            NumEmptyCycle=0 # number of cycles with no segment found
            H,W=ObjectMask.shape
            InsList = np.zeros([0, H,W]) #List of instances
            InstRank = [] # IOU score for the instances
            InstMap = np.zeros([H,W],int) # Map of all approved instances
            NInst=0
            OccupyMask=np.zeros([H,W],int) # Region that already been segmented
          #  ROIMask=ObjectMask.copy() # Region to be segmented
            NumPoints= int(340000 * 10/(H*W)) # Num instace points to guess per cycle
            if NumPoints<=1: NumPoints=2
#===============Generate instance map========================================================================================
            for cycle in range(NumSegCycles):
# .........................Generate input for the net................................................................
                PointerMask=np.zeros([NumPoints,H,W],dtype=float)
                ROI = np.ones([NumPoints, H, W], dtype=float)
                ImgList= np.ones([NumPoints, H, W,3], dtype=float)
                for i in range(NumPoints): # generate pointer mask
                        while(True):
                            px = np.random.randint(W)
                            py = np.random.randint(H)
                            if (ObjectMask[ py, px]) == 1 and OccupyMask[py,px]==0: break
                        PointerMask[i,py,px]=1
                        ImgList[i]=Img
                        ROI[i]=ObjectMask.copy()
                #*******************************************************************************************************
                #******************************************************************************************************
                # print("From readed")
                # for f in range(1):
                #     ImgList[f, :, :, 1] *= ObjectMask.astype(np.uint8)
                #     misc.imshow(ImgList[f])
                #     misc.imshow((ROI[f] * 2-OccupyMask + PointerMask[f] * 3).astype(np.uint8)*40)
                #******************************************************************************************************
                #******************************************************************************************************
    #=====================================Run Generator Net============================================================================================================================
                with torch.autograd.no_grad():
                    Prob, Lb = Generator.forward(Images=ImgList, Pointer=PointerMask, ROI=ROI, TrainMode=False)
                   # Prob, Lb, PredIOU, Predclasslist = MatNet.forward(Images=ImgList, Pointer=PointerMask,ROI=ROI,TrainMode=False,UseGPU=UseGPU, FreezeBatchNorm_EvalON=FreezeBatchNorm_EvalON)
                    Masks = Lb.data.cpu().numpy().astype(float)
                    PredIOU = Evaluator.forward(ImgList, Segment=Masks, ROI=ROI, TrainMode=False)
                    IOU = PredIOU.data.cpu().numpy().astype(float)
                # *******************************************************************************************************
                # ******************************************************************************************************
                # print("All Predicted")
                # for f in range(Masks.shape[0]):
                #         I = ImgList[f].copy()
                #         I[:, :, 1]=ImgList[f, :, :, 1] * Masks[f].astype(np.uint8)
                #         I[:, :, 2]=ImgList[f, :, :, 2] * (1 - ROI[f])
                #         I[:, :, 0]=ImgList[f, :, :, 0] * (1 - PointerMask[f])
                #         print(IOU[f])
                #         misc.imshow(I)
                #         misc.imshow((ROI[f] + Masks[f] * 2 + PointerMask[f] * 3).astype(np.uint8) * 40)
                # ******************************************************************************************************
                # ******************************************************************************************************
    #======================================Filter overlapping and low score predictions========================================================================================================
                IOUthresh=np.max(IOU)-0.1-cycle*0.1
                Accept=np.ones([NumPoints])
                for f in range(NumPoints):
                    SumMask=Masks[f].sum()
                    if IOU[f]<IOUthresh or (((Masks[f]*OccupyMask).sum()/(SumMask+0.001))>0.1 and cycle<20):
                           Accept[f]=0
                           continue


                    for i in range(NumPoints):
                        if i==f: continue
                        if IOU[f] > IOU[i] or Accept[i]==0: continue
                        fr=(Masks[i]*Masks[f]).sum()/(SumMask+0.00001)
                        if  (fr>0.05):
                                    Accept[f]=0
                                    break
                # *******************************************************************************************************
                # ******************************************************************************************************
                # print("IOU Thresh="+str(IOUthresh))
                # print("after first filtration")
                # for f in range(Masks.shape[0]):
                #         if Accept[f] == 0: continue
                #         I = ImgList[f].copy()
                #         I[:, :, 1] = ImgList[f, :, :, 1] * Masks[f].astype(np.uint8)
                #         I[:, :, 2] = ImgList[f, :, :, 2] * (1 - ROI[f])
                #         I[:, :, 0] = ImgList[f, :, :, 0] * (1 - PointerMask[f])
                #         print(IOU[f])
                #         misc.imshow(I)
                #         misc.imshow((ROI[f] + Masks[f] * 2 + PointerMask[f] * 3).astype(np.uint8) * 40)
                # # ******************************************************************************************************
    #===================================================Remove instace that overlap previously annotated regio or oyhrt segments with better score========================================================================================================================

                for f in range(NumPoints):
                    if Accept[f]==0: continue
                    OverLap = Masks[f] * OccupyMask
                    if (OverLap.sum() > 0):
                        Masks[f][OverLap>0] = 0
                    for i in range(NumPoints):
                        if Accept[i] == 0 or i==f or IOU[f]>IOU[i]: continue
                        OverLap=Masks[i]*Masks[f]
                        fr=(OverLap).sum()
                        if  (fr>0):  Masks[f][OverLap>0]=0

    # *******************************************************************************************************
    # ******************************************************************************************************
    #             print("after second filtration")
    #             for f in range(Masks.shape[0]):
    #                     if Accept[f] == 0: continue
    #                     I = ImgList[f].copy()
    #                     I[:, :, 1] = ImgList[f, :, :, 1] * Masks[f].astype(np.uint8)
    #                     I[:, :, 2] = ImgList[f, :, :, 2] * (1 - ROI[f])
    #                     I[:, :, 0] = ImgList[f, :, :, 0] * (1 - PointerMask[f])
    #                     print(IOU[f])
    #                     misc.imshow(I)
    #                     misc.imshow((ROI[f] + Masks[f] * 2 + PointerMask[f] * 3).astype(np.uint8) * 40)
    # ******************************************************************************************************
    #=============================Add selected  instances masks to final mask=======================================================================================================================================

                for f in range(NumPoints):
                        if Accept[f]==0: continue
                        NInst+=1
                      ##  InstCat.append(ClassPr[f])
                        InsList = np.concatenate([InsList,np.expand_dims(Masks[f],0)],axis=0)
                        InstRank.append(IOU[f])
                        InstMap[Masks[f]>0]=NInst
                        OccupyMask[Masks[f]>0]=1
    #=============================================================================================================================================================================================
                print("cycle"+str(cycle))


                if np.sum(Accept) == 0:
                    NumEmptyCycle += 1
                    if NumEmptyCycle > 15:
                        break
    #************************************************************************************************************************
    #***************************************************************************************************************************
                # print("ann map")
                # VizSegMapPr=np.expand_dims(InstMap,axis=2)
                # VizSegMapPr=np.concatenate([(VizSegMapPr*317)%255,(VizSegMapPr*17)%255,(VizSegMapPr*110)%255],axis=2).astype(np.uint8)
                # misc.imshow(VizSegMapPr)

    ##################################################################################################################

                if (OccupyMask.sum()/ ObjectMask.sum())>0.95: break

            # Img2 = Img.copy()
            # for i in range(InstMap.max()):
            #  Img2[:, :, 0][InstMap==i]+=i*30
            # Img2[:, :, 1] = 0.5 * Img2[:, :, 1] + 0.5 * (InstMap * 50).astype(np.uint8)
            # Img2[:, :, 2] = 0.5 * Img2[:, :, 2] + 0.5 * (InstMap * 93).astype(np.uint8)
            # print(NInst)
            # misc.imshow(InstMap * 30)
            # misc.imshow(cv2.resize(np.concatenate([Img, Img2], axis=1), (1000, 500)))

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
        print(str(i)+"File name")
        print(fname)
    #****************Resize if to big***********************************************************

        h=Mask.shape[1]
        w=Mask.shape[2]
       # if not fname=="ADE_val_00000404.jpg": continue
        MaxPixels=3000000
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
            ROIMask = np.expand_dims(ROIMask[0], axis=2)
            ROIMask=np.concatenate([ROIMask,ROIMask*0,ROIMask],axis=2)*200

            FullAnnGT=MatchIndexses(FullAnnGT,SegmentationMap)

            VizSegMapPr=np.expand_dims(SegmentationMap,axis=2)
            VizSegMapPr=np.concatenate([(VizSegMapPr*317)%255,(VizSegMapPr*17)%255,(VizSegMapPr*110)%255],axis=2).astype(np.uint8)
            FullAnnGT = np.expand_dims(FullAnnGT, axis=2)
            VizSegMapGT = np.concatenate([(FullAnnGT * 317) % 255, (FullAnnGT * 17) % 255, (FullAnnGT * 110) % 255],axis=2).astype(np.uint8)
           # misc.imshow(VizSegMap)
            Sep=np.ones([Img[0].shape[0],20,3],np.uint8)*255
            Overlay=np.concatenate([Img[0],Sep,ROIMask*0.7+Img[0]*0.3,Sep,VizSegMapGT * 0.7 + Img[0] * 0.3,Sep, VizSegMapPr*0.7+Img[0]*0.3],axis=1)
            Viz = np.concatenate([Img[0],Sep,ROIMask, Sep, VizSegMapGT , Sep, VizSegMapPr],axis=1)

            # misc.imshow(Overlay)
            # misc.imshow(Viz)
            cv2.imwrite(OutDir+"/"+str(i)+"_"+fname[:-4]+".png",Viz)
            cv2.imwrite(OutDir+"/"+str(i)+"_"+fname[:-4]+"Overlay.jpg", Overlay)

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



