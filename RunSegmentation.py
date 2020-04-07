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
InImage=r"Example/Input/1_IMG.jpg" #  Input image
InObjectMask=r"Example/Input/1_ObjectMask.png" # Input object region mask
OutAnnotationFile=r"Example/Output/PredictedAnnotation.png" # Output file where the annotation will
MinIOUThresh=-10 # Min threshold for segment to be accepted
MaxOverlap=0.3 # maximum overlap between generated segment and previously accepted segments


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
#***************************************Load images***********************************************************************************************************
Img=cv2.imread(InImage)[..., ::-1]
ObjectMask=cv2.imread(InObjectMask,0)

Img=np.expand_dims(Img,0)
ObjectMask=np.expand_dims(ObjectMask,0)
UnsegmentedRegion=ObjectMask.copy()
SegmentationMap=np.zeros([Img.shape[1],Img.shape[2]])
################################################Create statitics list##############################################################################################################################################################################################




#****************************Split object into part main function***********************************************************************************
SegmentationMap=soip.SplitObjectToParts(Img[0], ObjectMask[0],Generator,Evaluator,MinIOUThresh,MaxOverlap)
# ***************************************Find the predicted mask best match the GT mask in terms of IOU***********************************************************************************************************



Img[0]=Img[0][..., ::-1]
ObjectMask = np.expand_dims(ObjectMask[0], axis=2)
ObjectMask=np.concatenate([ObjectMask,ObjectMask*0,ObjectMask],axis=2)*200


VizSegMapPr=np.expand_dims(SegmentationMap,axis=2)
VizSegMapPr=np.concatenate([(VizSegMapPr*317)%255,(VizSegMapPr*17)%255,(VizSegMapPr*110)%255],axis=2).astype(np.uint8)

# misc.imshow(VizSegMap)
Sep=np.ones([Img[0].shape[0],20,3],np.uint8)*255
#Overlay=np.concatenate([Img[0],Sep,ObjectMask*0.7+Img[0]*0.3, Sep, VizSegMapPr*0.7+Img[0]*0.3],axis=1)
Viz = np.concatenate([Img[0],Sep,ObjectMask,  Sep, VizSegMapPr],axis=1)
cv2.imwrite(OutAnnotationFile,Viz)
print("Annoatation saved to "+OutAnnotationFile)
#cv2.imshow("Overlay",Overlay.astype(np.uint8)
cv2.imshow("Segment",Viz)
cv2.waitKey()


