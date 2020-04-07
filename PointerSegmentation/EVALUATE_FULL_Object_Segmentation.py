# Evaluate the accuracy for segmentation of the full object require fully trained net and GT annotation

#..............Imports..................................................................
import ConvertLabelToOneHotEncoding
import os
import torch
import numpy as np
import scipy.misc as misc
#import CocoPanoptic_Reader as Data_Reader
import ReaderParts as Data_Reader
#import DeepLab_FCN_NetModel as NET_FCN
import FCN_NetModel as NET_FCN # The net Class
import cv2
#modelDir="logs/"
##################################Input paramaters#########################################################################################
#.................................Main Input parametrs...........................................................................................
Trained_model_path="logs/1200000.torch"  # Trained model weight
# AnnDir=r"/media/sagi/2T/Data_zoo/ADE20K_2016_07_26/LabelConvertedValidation2/" # GT annotation of object parts
# ImageDir=r"/media/sagi/2T/Data_zoo/ADE20k_PARTS/ImageConvertedValidation//" # Images folder

AnnDir="Example/Training/Anns/"
ImageDir="Example/Training/Images//"

# AnnDir="/media/sagi/2T/GES_PARTS/PascalPartData/Eval/Label/"
# ImageDir="/media/sagi/2T/GES_PARTS/PascalPartData/Eval/Image//"


UseCrop=True # Crop the object region from the image
WriteAnn=True # Flag do you wish to save the output of each annotation into file
if WriteAnn:
   OutDir=r"/media/sagi/2T/Data_zoo/ResultsPascalfam/"
   if not os.path.isdir(OutDir): os.mkdir(OutDir)
# fl=open("EvalResults.txt","w")
# fl.write("")
# fl.close()
##################################Make sre matchjng GT and pred segments will have same index. Mainly for conisstent visualization of GT and prediction######################################################################
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
#############################################################################################################################################################33




#---------------------Create and Initiate net-----------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained
Net=Net.cuda()
#for Trained_model_path in Trained_model_paths: # Evaluate all models in the model folder
Net.load_state_dict(torch.load(Trained_model_path))
Net.eval()
Net.half()
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader=Data_Reader.Reader(ImageDir,AnnDir,TrainingMode=False,InverseSelection=False)# MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,TrainingMode=False)
print("Start Evaluating "+Trained_model_path)
Siou={}
Sprec={}
Srecall={}
Sn={}

#------------------------------------Evaluation loop---------------------------------------------------------------------------------
for i in range(4000):
        #Img, Mask, PointerMask, ROIMask, CatID, sy, sx = Reader.LoadSingle(ByClass=True)
        Img, Mask, PointerMask, ROIMask, FullAnnGT, fname, PartLevel, PartClass, Class = Reader.LoadSingle()
        print(str(i) + "File name")
        print(fname)
        if ("ADE_val_00001017.jpg" in fname) or ("ADE_val_00000322.jpg" in fname): continue
# ****************Crop object region***********************************************************
        if UseCrop: #  Crop the object region from the image
            d, h, w = Mask.shape
            x, y, wb, hb = cv2.boundingRect((ROIMask[0] > 0).astype(np.uint8))
            # if hb<384:
            d = np.max([424 - hb, 80])
            y1 = int(np.max([y - (d / 2), 0]))
            y2 = int(np.min([y1 + np.max([hb + 40, 424]), h - 1]))

            d = np.max([424 - wb, 80])
            x1 = int(np.max([x - (d / 2), 0]))
            x2 = int(np.min([x1 + np.max([wb + 40, 424]), w - 1]))

            d, h, w = Mask.shape
            Img = Img[:, y1:y2, x1:x2]
            FullAnnGT = FullAnnGT[y1:y2, x1:x2]
            Mask = Mask[:, y1:y2, x1:x2]
            ROIMask = ROIMask[:, y1:y2, x1:x2]
            PointerMask= PointerMask[:, y1:y2, x1:x2]
# ***********************************************************************************************************************************************
    #***********************************************************************************************************************************************
        # for f in range(Img.shape[0]):
        #     Img[f, :, :, 0] *= 1-Mask[f]
        #     Img[f, :, :, 1] *= ROIMask[f]
        #     Img[f, :, :, 1] *= FullAnn
        #     misc.imshow((ROIMask[f] + Mask[f] * 2 + PointerMask[f] * 3).astype(np.uint8)*40)
        #     misc.imshow(Img[f])
        # print(ROIMask.shape)
#***************************************Start loop tht segment the entire objects into parts  by part and pick the segment  best fitted to the target segment in term of IOU***********************************************************************************************************
        SegmentList=np.zeros([100,Img.shape[1],Img.shape[2]],dtype=np.float32) # List of all the parts masks (Parts can overlap)
        PrecisionList=np.zeros([100]) # Precision of each part mask and the target mask
        IOUList = np.zeros([100]) # IOU
        RecallList = np.zeros([100]) # Object region tht reamin unsegmented

        UnsegmentedRegion=ROIMask.copy()
        SegmentationMap=np.zeros([Img.shape[1],Img.shape[2]])
        for ff in range(99): # Main segmenting loop find all the parts of the object in the ROI region


            PointerMask=Reader.GeneratePointermask(UnsegmentedRegion[0])# Create Pointer mask
            PointerMask=np.expand_dims(PointerMask,0)
            with torch.no_grad():
                Prob, Lb=Net.forward(Images=Img,Pointer=PointerMask,ROI=ROIMask,TrainMode=False) # Run net inference and get prediction
            Pred=Lb.cpu().data.numpy()*UnsegmentedRegion

            UnsegmentedRegion[Pred>0]=0 # Update ROI mask
#------------Calculate intersection over union between prediction and GT--------------------------------------------------
            Inter=(Pred*Mask).sum()
            Gs=Mask.sum()
            Ps=Pred.sum()
            IOU=Inter/(Gs+Ps-Inter)
            Precision=Inter/(Ps+0.0001)
            Recall=Inter/(Gs+0.00001)

            PrecisionList[ff]=Precision
            IOUList[ff]=IOU
            RecallList[ff]=Recall
            SegmentList[ff]=Pred[0]
            SegmentationMap[Pred[0]>0]=ff+1
            if (UnsegmentedRegion.sum()/ROIMask.sum())<0.04: break


        ind=np.argmax(IOUList)
        IOU=IOUList[ind]
        Precision=PrecisionList[ind]
        Recall=RecallList[ind]
        Pred[0]=SegmentList[ind]

    #********************Save visualization of output and GT into file******************************************************************************
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

    #************************************Add to general statistics**********************************************************************************************************************************
        if not Class in Siou:
            Siou[Class] = IOU
            Sprec[Class] = Precision
            Srecall[Class] = Recall
            Sn[Class] = 1

        Siou[Class] += IOU
        Sprec[Class] += Precision
        Srecall[Class] += Recall
        Sn[Class]+=1

    #***********************************convert to np.array*******************************************************************************************************************

        SiouAr= np.array(list(Siou.values())) # iou
        SprecAr= np.array(list(Sprec.values())) # prec
        SrecallAr= np.array(list(Srecall.values())) # recall
        SnAr= np.array(list(Sn.values())) # number of of cases

        #*******************************save final statics***********************************************************************************************************************
        Iou=(SiouAr/(SnAr+0.000001)).sum()/(SnAr>0).sum()
        Precision=(SprecAr/(SnAr+0.000001)).sum()/(SnAr>0).sum()
        Recall=(SrecallAr/(SnAr+ 0.000001)).sum()/(SnAr>0).sum()
        txt="\n "+Trained_model_path+"Num Class=\t"+str((SnAr>0).sum())+"\tNum="+str(SnAr.sum())+"\tIOU="+str(Iou)+"\tPrecission="+str(Precision)+"\tRecall="+str(Recall)
        print(txt)
    # fl=open("logs/EvalResults.txt","a")
    # fl.write(txt)
    # fl.close()





# Results Unfamiliar logs//1200000.torchNum Class=	38	Num=4038	IOU=0.5047560520268203	Precission=0.6983280529896474	Recall=0.700490212312625



