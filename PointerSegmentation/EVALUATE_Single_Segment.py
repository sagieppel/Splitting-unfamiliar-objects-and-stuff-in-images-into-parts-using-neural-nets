# Evaluate the accuracy of the net on segmenting single part of the objects
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

##################################Input paramaters#########################################################################################
#.................................Main Input parametrs...........................................................................................

#AnnDir=r"/media/sagi/2T/Data_zoo/ADE20K_2016_07_26/LabelConvertedValidation2/" # GT annotation of object parts
#ImageDir=r"/media/sagi/2T/Data_zoo/ADE20k_PARTS/ImageConvertedValidation//" # Images folder
#AnnDir="/media/sagi/2T/GES_PARTS/PascalPartData/Eval/Label/"
#ImageDir="/media/sagi/2T/GES_PARTS/PascalPartData/Eval/Image//"
AnnDir="Example/Training/Anns/"
ImageDir="Example/Training/Images//"


Trained_model_path="logs/1200000.torch"  # Trained model weight # Path to traine model
fl=open("EvalResults.txt","w") # output file
fl.write("")
fl.close()
##-----------------------------------List of model to evaluae----------------------------------------------------------------------------




# MaxBatchSize=7 # Max images in batch
# MinSize=250 # Min image Height/Width
# MaxSize=1000# Max image Height/Width



#---------------------Create and Initiate net-----------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained
Net=Net.cuda()
#for Trained_model_path in Trained_model_paths: # Evaluate all models in the model folder
Net.load_state_dict(torch.load(Trained_model_path))
Net.eval()
Net.half()
    #----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader=Data_Reader.Reader(ImageDir,AnnDir,TrainingMode=False,InverseSelection=True)# MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,TrainingMode=False)


print("Start Evaluating "+Trained_model_path)
Siou={}
Sprec={}
Srecall={}
Sn={}




#------------------------------------Evaluation loop---------------------------------------------------------------------------------
for i in range(8000):
        #Img, Mask, PointerMask, ObjectMask, CatID, sy, sx = Reader.LoadSingle(ByClass=True)
      #  Img, Mask, PointerMask, ObjectMask, fname, PartLevel, PartClass, Class = Reader.LoadSingle()
        Img, Mask, PointerMask, ObjectMask, FullAnnGT, fname, PartLevel, PartClass, Class = Reader.LoadSingle()
    #***********************************************************************************************************************************************
        # for f in range(Img.shape[0]):
        #     Img[f, :, :, 0] *= 1-Mask[f]
        #     Img[f, :, :, 1] *= ObjectMask[f]
        #
        #     misc.imshow((ObjectMask[f] + Mask[f] * 2 + PointerMask[f] * 3).astype(np.uint8)*40)
        #     misc.imshow(Img[f])
        # print(ObjectMask.shape)
    #**************************************************************************************************************************************************


        with torch.no_grad():
            Prob, Lb=Net.forward(Images=Img,Pointer=PointerMask,ROI=ObjectMask,TrainMode=False) # Run net inference and get prediction
        Pred=Lb.cpu().data.numpy()
        Inter=(Pred*Mask).sum()
        Gs=Mask.sum()
        Ps=Pred.sum()
        if Gs.sum()<100: continue
        IOU=Inter/(Gs+Ps-Inter)
        Precision=Inter/(Ps+0.0001)
        Recall=Inter/Gs
        if not Class in Siou:
            Siou[Class] = IOU
            Sprec[Class] = Precision
            Srecall[Class] = Recall
            Sn[Class] = 1

        Siou[Class] += IOU
        Sprec[Class] += Precision
        Srecall[Class] += Recall
        Sn[Class]+=1
    #******************************************************************************************************************************************************
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
        txt="\n "+Trained_model_path+"Num Class=\t"+str((SnAr>0).sum())+"\tNum="+str(SnAr.sum())+"\tIOU="+str(Iou)+"\tPrecission="+str(Precision)+"\tRecall="+str(Recall)
        print(txt)
        fl=open("logs/EvalResults.txt","a")
        fl.write(txt)
        fl.close()









