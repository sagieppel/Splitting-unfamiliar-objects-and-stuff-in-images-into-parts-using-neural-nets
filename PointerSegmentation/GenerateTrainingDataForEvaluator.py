
#Apply the pointer net to generate training data  for the evaluator (GES net)
#...............................Imports..................................................................
#import ConvertLabelToOneHotEncoding
import os
import torch
import numpy as np
import cv2
import scipy.misc as misc
#import CocoPanoptic_Reader as Data_Reader
import ReaderParts as Data_Reader
#import DeepLab_FCN_NetModel as NET_FCN
import FCN_NetModel as NET_FCN # The net Class

########################input model dir############################
modelDir="logs/" # all models in this folder will be used to generate data
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
#.................................Main Input Data location..........................................................................................

# AnnDir="/media/sagi/2T/GES_PARTS/PascalPartData/Train/Label/"
# ImageDir="/media/sagi/2T/GES_PARTS/PascalPartData/Train/Image//"
#
AnnDir="Example/Training/Anns/"
ImageDir="Example/Training/Images//"
#-----------------------------OutPut dir-----------------------------------------------------------------------
#OutDir="../ValidationDataForEvaluator/" #"../TrainingDataForEvaluator/"
OutDir="../TrainingDataForEvaluator/"

OutAnnDir=OutDir+"/Ann/"
OutImgDir=OutDir+"/Img//"
if not os.path.exists(OutDir): os.mkdir(OutDir)
if not os.path.exists(OutImgDir): os.mkdir(OutImgDir)
if not os.path.exists(OutAnnDir): os.mkdir(OutAnnDir)
#=========================================Generate list f models==================================================


Trained_model_files=[]
for Name in os.listdir(modelDir):
    if ".torch" in Name:
        Trained_model_files.append(Name)
Trained_model_files.sort()

# ----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------

Reader=Data_Reader.Reader(ImageDir,AnnDir,TrainingMode=False, Suffle=True,InverseSelection=False, UseAllClaseses=True)# MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,TrainingMode=False)
#---------------------Create and Initiate net ------------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained
Net=Net.cuda()
Trained_model=Trained_model_files[np.random.randint(len(Trained_model_files))]
Reader.Reset()



Net.load_state_dict(torch.load(modelDir+"/"+Trained_model))
Net.eval()
Net.half()
fif=0

#----------------------------------------------------main data generation loop------------------------------------------------------------------------------------------------------------------------------
while (True):
    fif+=1
    #if fif>1200000: break
    print(fif)
    x=open(OutDir+"/Count.txt","w")
    x.write(str(fif))
    x.close()
    if (fif%1000==0): # Randomly replace net every 1000 steps
        Trained_model_files = []
        for Name in os.listdir(modelDir):
            if ".torch" in Name:
                Trained_model_files.append(Name)
        Trained_model_files.sort()
        Trained_model = Trained_model_files[np.random.randint(len(Trained_model_files))]
        Net.load_state_dict(torch.load(modelDir + "/" + Trained_model))
        print("Loading model "+Trained_model)
        Net.eval()
        Net.half()

#while (Reader.Epoch<1):
    print(Reader.Clepoch.min())
    Img, GTMask, PointerMask, ROIMask,fname, PartLevel, PartClass, Class = Reader.LoadSingleForGeneration()

    #       Img, GTMask, PointerMask, ROIMask, CatID,ImName, sy, sx = Reader.LoadSingleForGeneration(ByClass=False,Augment=False)
#***********************************************************************************************************************************************
   # GTMask*= 0
   #  for f in range(Img.shape[0]):
   #  #    misc.imshow(Img[f])
   #      Img[f, :, :, 0] *= 1 - GTMask[f]
   #      Img[f, :, :, 1] *= 1 - PointerMask[f]
   #      Img[f, :, :, 2] *= 1 - ROIMask[f]
   #
   #      misc.imshow((ROIMask[f] + GTMask[f] * 2 + PointerMask[f] * 3).astype(np.uint8)*40)
   #      misc.imshow(Img[f])
   #  print(ROIMask.shape)
#*****************************************Run prediction and generate data*********************************************************************************************************


    with torch.no_grad():
        Prob, Lb=Net.forward(Images=Img,Pointer=PointerMask,ROI=ROIMask,TrainMode=False) # Run net inference and get prediction
    Pred=Lb.cpu().data.numpy()
    Inter=(Pred*GTMask).sum()
    Gs=GTMask.sum()
    Ps=Pred.sum()
    IOU=Inter/(Gs+Ps-Inter)
    Precision=Inter/Ps
    Recall=Inter/Gs
    fname=fname[:-4]+"#IOU#"+str(IOU)+"#Precision#"+str(Precision)+"#Recall#"+str(Recall)+"#Class#"+str(Class)+"#PartClass#"+str(PartClass)+"#PartLevel#"+str(PartLevel)+"#RandID#"+str(np.random.randint(0,1000000000))
    print(fname)
  #  cv2.imwrite(OutGTDir+"/"+fname,GTMask[0].astype(np.uint8))
    Cn=np.concatenate([np.expand_dims(Pred[0].astype(np.uint8),2),np.expand_dims(GTMask[0].astype(np.uint8),2),np.expand_dims(ROIMask[0].astype(np.uint8),2)],axis=2)
    cv2.imwrite(OutImgDir + "/" + fname + ".jpg", Img[0][..., ::-1])
    cv2.imwrite(OutAnnDir + "/" + fname + ".png", Cn)

  #  misc.imshow(Cn*100)

#******************************************************************************************************************************************************
    # print()
    # print("Precision=" + str(Precision))
    # print("Recall=" + str(Recall))
    #
    # Img[0, :, :, 0] *= 1-GTMask[0]
    # Img[0, :, :, 1] *= 1-Pred[0]
    # misc.imshow(Img[0])

#******************************************************************************************************************************************************
x = open(OutDir + "/Finished.txt", "w")
x.close()











