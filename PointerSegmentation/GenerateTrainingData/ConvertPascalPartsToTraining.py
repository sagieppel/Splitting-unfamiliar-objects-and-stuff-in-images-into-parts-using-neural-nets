#Divide images to class sets
import os
import numpy as np
import matplotlib.pylab as plt
from skimage.io import imread
#from VOClabelcolormap import color_map
from PascalPartsAnno import ImageAnnotation
import scipy.misc as misc
import os
from shutil import copyfile
import cv2
##############input parameters############################################################################
# PascalPartLabelDir="/media/sagi/2T/GES_PARTS/PascalPartData/OriginalFormat/Eval/Label/" # Input pascal annotatation folder
# PascalImageDir="/media/sagi/2T/GES_PARTS/PascalPartData/OriginalFormat/Eval/Image/" # Input Pascal Image folder
PascalPartLabelDir="/media/sagi/2T/Data_zoo/PascalParts/PascalPArtsNew/trainval/Annotations_Part/"# Input pascal annotatation folder
PascalImageDir="/media/sagi/2T/Data_zoo/PascalParts/PascalPArtsNew/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/"# Input Pascal Image folder
OutDir="/media/sagi/2T/GES_PARTS/PascalPartData//Eval/" # Output converted annotation dir
OutPascalPartLabelDir=OutDir+"/Label//"
OutImgDir=OutDir+"/Image//"
##########################################################################################################
if not os.path.exists(OutDir): os.makedirs(OutDir)
if not os.path.exists(OutPascalPartLabelDir): os.makedirs(OutPascalPartLabelDir)
if not os.path.exists(OutImgDir): os.makedirs(OutImgDir)


ClassName = {1: 'aeroplane',
           2: 'bicycle',
           3: 'bird',
           4: 'boat',
           5: 'bottle',
           6: 'bus',
           7: 'car',
           8: 'cat',
           9: 'chair',
           10: 'cow',
           11: 'table',
           12: 'dog',
           13: 'horse',
           14: 'motorbike',
           15: 'person',
           16: 'pottedplant',
           17: 'sheep',
           18: 'sofa',
           19: 'train',
           20: 'tvmonitor'}


#an.im The image rgb
#an.cls_mask # standart annotation mask one number per class (only for classes that have parts)
#an.inst_mask # Instasnce segmentation mask fdifferent number for each instance only for class that appear in the segmentation
#an.part_mask # segmentation mask with parts of all objects note that each segment in cls_mask will have its own set of part class which may overlap


###########################################################################################################################
# Files = []
# Files += [each for each in os.listdir(PascalPartLabelDir) if each.endswith('.mat')]  # Get list of training images
# for i in range(30):
#     if not os.path.exists(OutDir+str(i)+"/Annotation/"): os.makedirs(OutDir+str(i)+"/Annotation/")
#     if not os.path.exists(OutDir + str(i) + "/Images/"): os.makedirs(OutDir + str(i) + "/Images/")
for  Fn in  os.listdir(PascalPartLabelDir):
    if not '.mat' in Fn: continue
    an = ImageAnnotation(PascalImageDir+Fn[:-3]+"jpg",PascalPartLabelDir+Fn)
    im=misc.imread(PascalImageDir + Fn[:-3] + "jpg")
#------------go over all objects in the image-------------------------------------------------
    for i in np.unique(an.inst_mask):
        if i==0: continue

#-----------------Extract object mask
        ObjMask=(an.inst_mask == i).astype(np.uint8) #Found mask of specific object intance


#------------------Find object class--------------------------------------
        cls=-1 # class
        ClassMask=ObjMask * an.cls_mask
        for cl in np.unique(ClassMask):
                    if cl == 0: continue
                    fr = np.sum(cl==ClassMask) / np.sum(ObjMask)
                    if fr>0.6:
                        cls=cl
                        break
        print(ClassName[cl])
        # misc.imshow((ObjMask).astype(np.uint8) * 255)
        # misc.imshow(im)
        if cls==-1:
            continue
#------------------------------Go Over all parts in the image-------------------------------------------------
        PartsMap=an.part_mask*ObjMask
        AllParts = np.zeros(PartsMap[:, :].shape, np.uint8)
 #----------Create sequential part mask to an object--------------------------------
        for prt in np.unique(PartsMap):
            if prt==0: continue
            prtMask=(PartsMap==prt).astype(np.uint8)
            AllParts[prtMask>0]=AllParts.max()+1
#---------------Save each object part instance in a differen mask
        prIns=0
        for prt in np.unique(PartsMap):
            if prt == 0: continue
            PartMask = (PartsMap == prt).astype(np.uint8)
            prIns+=1





            outLabelpath = OutPascalPartLabelDir+ "/" +  Fn[:-3] + "_PartsLevel_0_PartInstance_" + str(prIns) + "_PartClass_" + str(int(prt)) + "_TopClass_" + str(int(cls)) + ".png" # assign name for class note that the part class and level are madeup
            Label = np.concatenate([np.expand_dims(PartMask * 255, axis=2), np.expand_dims(ObjMask * 255, axis=2), np.expand_dims((AllParts).astype(np.uint8), axis=2)], axis=2)
            cv2.imwrite(outLabelpath, Label)
            cv2.imwrite(OutImgDir + "/" + Fn[:-3] + ".jpg", im[..., ::-1])





            # misc.imshow(Label[:,:,2]*10)
            # misc.imshow(Label)
            # misc.imshow(im)
            # misc.imshow(prtMask *100)
            # misc.imshow(PartsMap * 10)



        # if (an.cls_mask.astype(np.int8)==i).sum():
        #
        #     copyfile(PascalImageDir + Fn[:-3] + "jpg", OutDir + str(i) +  "/Images/"+ Fn[:-3] + "jpg")
        #     copyfile(PascalPartLabelDir + Fn, OutDir+str(i)+"/Annotation/" + Fn)



