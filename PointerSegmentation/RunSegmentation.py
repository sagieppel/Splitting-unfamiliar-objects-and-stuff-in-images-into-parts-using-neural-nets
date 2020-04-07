# Get an image an a mask of the region of an object in the image and segment the object into parts
#..............Imports..................................................................
import ConvertLabelToOneHotEncoding
import os
import torch
import numpy as np
#import scipy.misc as misc
#import CocoPanoptic_Reader as Data_Reader
import ReaderParts as Reader
#import DeepLab_FCN_NetModel as NET_FCN
import FCN_NetModel as NET_FCN # The net Class
import cv2
#modelDir="logs/"
##################################Input paramaters#########################################################################################
#.................................Main Input parametrs...........................................................................................
Trained_model_path="logs/1200000.torch" # Trained model weight
InImage=r"Example/Input/1_IMG.jpg" #  Input image
InObjectMask=r"Example/Input/1_ObjectMask.png" # Input object region mask
OutAnnotationFile=r"Example/Output/PredictedAnnotation.png" # Output file where the annotation will
#---------------------Create and Initiate net-----------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained
Net=Net.cuda()
Net.load_state_dict(torch.load(Trained_model_path))
Net.eval()
Net.half()
#***************************************Load images***********************************************************************************************************
Img=cv2.imread(InImage)[..., ::-1]
ObjectMask=cv2.imread(InObjectMask,0)

Img=np.expand_dims(Img,0)
ObjectMask=np.expand_dims(ObjectMask,0)
UnsegmentedRegion=ObjectMask.copy()
SegmentationMap=np.zeros([Img.shape[1],Img.shape[2]])
for ff in range(100): # Main segmenting loop find all the parts of the object in the ROI region'
    print("Segment Num="+str(ff))
#...........Generate pointer mask........................
    PointerMask = np.zeros(ObjectMask.shape, dtype=np.float)
    while (True):
        x = np.random.randint(0, ObjectMask.shape[2])
        y = np.random.randint(0, ObjectMask.shape[1])
        if UnsegmentedRegion[0,y, x] > 0:
            PointerMask[0,y, x] = 1
            break

#.................Run prediction and add to annotation mask...........................................
    #PointerMask=np.expand_dims(PointerMask,0)
    with torch.no_grad():
        Prob, Lb=Net.forward(Images=Img,Pointer=PointerMask,ROI=ObjectMask,TrainMode=False) # Run net inference and get prediction
    Pred=Lb.cpu().data.numpy()#*UnsegmentedRegion
    s1=Pred.sum()
    Pred*=UnsegmentedRegion
    if Pred.sum()/s1<0.7: continue
    UnsegmentedRegion[Pred>0]=0 # Update ROI mask
    SegmentationMap[Pred[0]>0]=ff+1


#********************Save visualization of output  into file******************************************************************************
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

#cv2.imwrite(OutAnnotationFile, Overlay)

#************************************Add to general statistics**********************************************************************************************************************************



