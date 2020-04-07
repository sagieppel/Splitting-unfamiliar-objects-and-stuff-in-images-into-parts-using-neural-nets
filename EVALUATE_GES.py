# Evaluate the accuracy of the generator evaluator system for parts segmentation
# Use the trained pointer generator and evaluator to annotate object and compare it to GT annnotation
#..............Imports..................................................................
import os
import torch
import PointerSegmentation.ReaderParts as Data_Reader
import Evaluator.NetModel as EvaluatorNet
import numpy as np
import PointerSegmentation.FCN_NetModel as GeneratorNet # The net Class
import cv2
import scipy.misc as misc
import SplitObjectIntoParts as soip
#modelDir="logs/"


#.................................Main Input parametrs...........................................................................................


# MaskDir=r"/media/sagi/2T/Data_zoo/ADE20K_2016_07_26/LabelConvertedValidation2/" #GT annotation folder
# ImageDir=r"/media/sagi/2T/Data_zoo/ADE20k_PARTS/ImageConvertedValidation//" # Images folder
# ImageDir="/media/sagi/2T/GES_PARTS/PascalPartData/Eval/Image/"# GT annotation folder
# MaskDir="/media/sagi/2T/GES_PARTS/PascalPartData/Eval/Label/" # Images folder
MaskDir="PointerSegmentation/Example/Training/Anns/"
ImageDir="PointerSegmentation/Example/Training/Images//"
MinIOUThresh=-10 # Min threshold for segment to be accepted
MaxOverlap=0.3 # maximum overlap between generated segment and previously accepted segments

CropObjectRegion=True # Crop the region of the object from the image and use only this region as input

WriteAnn=True # Write predictions into file
if WriteAnn:
   OutDir=r"Example/OutEval/" #folder were the predicted annotations will be saved
   if not os.path.isdir(OutDir): os.mkdir(OutDir)
#
# Generator_model_path = "PointerSegmentation/logs/200000.torch" #Trained mode for generator
# Evaluator_model_path = "Evaluation/logs/130000.torch"# Trained model for evaluator

Generator_model_path = "PointerSegmentation/logs/1200000.torch" #Trained mode for generator
Evaluator_model_path = "Evaluator/logs/600000.torch"# Trained model for evaluator
#IOUthresh=0.8
###################################Loading nets###############################################################
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
################################################Create annotation reader########################################################################################

#----------------------------------------Create reader for data set (use the pointer net reader)--------------------------------------------------------------------------------------------------------------
Reader=Data_Reader.Reader(ImageDir,MaskDir,Generator,Evaluator,TrainingMode=False,InverseSelection=False,UseAllClaseses=False)# MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,TrainingMode=False)


################################################Create statitics list##############################################################################################################################################################################################



print("Start Evaluating ")
Siou={}
Sprec={}
Srecall={}
Sn={}


Con=True

#------------------------------------Mmain Evaluation loop---------------------------------------------------------------------------------
for i in range(100000):

        #Img, Mask, PointerMask, ObjectMask, CatID, sy, sx = Reader.LoadSingle(ByClass=True)
        Img, Mask, PointerMask, ObjectMask, FullAnnGT, fname, PartLevel, PartClass, Class = Reader.LoadSingle() # read annotation
        if Class in Sn and Sn[Class]>300: continue
        if (Mask.sum()/ObjectMask.sum())<0.01 or Mask.sum()<100: continue#***********************************************************
        print(str(i)+"File name")
        print(fname)
    #    if ("ADE_val_00001017.jpg" in fname) or ("ADE_val_00000322.jpg" in fname): continue
    #****************Crop object region***********************************************************
        if CropObjectRegion:
            d,h,w=Mask.shape
            x, y, wb, hb = cv2.boundingRect((ObjectMask[0]>0).astype(np.uint8))
            #if hb<384:
            d=np.max([324-hb,80])
            y1=int(np.max([y-(d/2),0] ))
            y2=int(np.min([y1+np.max([hb+40,324]),h-1]))

            d = np.max([324 - wb, 80])
            x1 = int(np.max([x - (d / 2), 0]))
            x2 = int(np.min([x1 + np.max([wb+40, 324]), w - 1]))

            d, h, w = Mask.shape
            Img=Img[:,y1:y2,x1:x2]
            FullAnnGT = FullAnnGT[y1:y2,x1:x2]
            Mask= Mask[:,y1:y2,x1:x2]
            ObjectMask= ObjectMask[:,y1:y2,x1:x2]
    #***********************************************************************************************************************************************
        # for f in range(Img.shape[0]):
        #     Img[f, :, :, 0] *= 1-Mask[f]
        #     Img[f, :, :, 1] *= 1-ObjectMask[f]
        #   #  Img[f, :, :, 1] *= FullAnnGT
        #     misc.imshow((ObjectMask[f] + Mask[f]).astype(np.uint8)*40)
        #     misc.imshow(Img[f])
        # print(ObjectMask.shape)
        SegmentTop=np.zeros([Img.shape[1],Img.shape[2]],dtype=np.float32) # List of all the parts masks (Parts can overlap)
        PrecisionTop=0 # Precision of each part mask and the target mask
        IOUTop = 0 # IOU
        RecallTop = 0 # Object region tht reamin unsegmented

#****************************Split object into part main function***********************************************************************************
        SegmentationMap=soip.SplitObjectToParts(Img[0], ObjectMask[0],Generator,Evaluator,MinIOUThresh,MaxOverlap)
# ***************************************Find the predicted mask best match the GT mask in terms of IOU***********************************************************************************************************



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

    #********************Save segmentation map to file segmentation map******************************************************************************
        if WriteAnn:
            Img[0] = Img[0][..., ::-1]
            ObjectMask = np.expand_dims(ObjectMask[0], axis=2)
            ObjectMask=np.concatenate([ObjectMask*220,ObjectMask*1,ObjectMask*220],axis=2)

            FullAnnGT=soip.MatchIndexses(FullAnnGT,SegmentationMap)

            VizSegMapPr=np.expand_dims(SegmentationMap,axis=2)
            VizSegMapPr=np.concatenate([(VizSegMapPr*317)%255,(VizSegMapPr*17)%255,(VizSegMapPr*110)%255],axis=2).astype(np.uint8)
            FullAnnGT = np.expand_dims(FullAnnGT, axis=2)
            VizSegMapGT = np.concatenate([(FullAnnGT * 317) % 255, (FullAnnGT * 17) % 255, (FullAnnGT * 110) % 255],axis=2).astype(np.uint8)
           # misc.imshow(VizSegMap)
            Sep=np.ones([Img[0].shape[0],20,3],np.uint8)*255
            Overlay=np.concatenate([Img[0],Sep,ObjectMask*0.7+Img[0]*0.3,Sep,VizSegMapGT * 0.7 + Img[0] * 0.3,Sep, VizSegMapPr*0.7+Img[0]*0.3],axis=1)
            Viz = np.concatenate([Img[0],Sep,ObjectMask, Sep, VizSegMapGT , Sep, VizSegMapPr],axis=1)

            I1 = Img[0].copy()
            I1[:, :][ObjectMask>0 ]=0
            Overlay2 = np.concatenate([Img[0], Sep, ObjectMask+ I1*0.5, Sep, VizSegMapGT + I1*0.5 , Sep,VizSegMapPr + I1*0.5], axis=1)



            # misc.imshow(Overlay)
            # misc.imshow(Viz)
            nw=1600
            nh=int(nw/Viz.shape[1]*Viz.shape[0])


            cv2.imwrite(OutDir+"/"+str(i)+"_"+fname[:-4]+".png",cv2.resize(Viz,(nw,nh)))
            cv2.imwrite(OutDir+"/"+str(i)+"_"+fname[:-4]+"Overlay.jpg",cv2.resize(Overlay,(nw,nh)))
            cv2.imwrite(OutDir + "/" + str(i) + "_" + fname[:-4] + "Overlay2.png",cv2.resize(Overlay2,(nw,nh)))

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



