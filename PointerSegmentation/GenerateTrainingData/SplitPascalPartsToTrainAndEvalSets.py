# Split pascal parts  data to training and evaluation

import os
from shutil import copyfile
import sys

TxtFileDirs="/media/sagi/2T/Data_zoo/PascalParts/PascalPArtsNew/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/"
LabelDir="/media/sagi/2T/Data_zoo/PascalParts/PascalPArtsNew/trainval/Annotations_Part/"
ImageDir="/media/sagi/2T/Data_zoo/PascalParts/PascalPArtsNew/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/"
OutDir="/media/sagi/2T/GES_PARTS/PascalPartData"
Files=[]
Files += [each for each in os.listdir(TxtFileDirs) if each.endswith('_train.txt')] # Get list of training images
if not os.path.exists(OutDir): os.makedirs(OutDir)
#............................................................................................................

OutEvalDir=OutDir+"/Eval/"
OutTrainDir=OutDir+"/Train/"

OutEvalImageDir = OutEvalDir + "/Image/"
OutEvalLabelDir = OutEvalDir + "/Label/"

OutTrainImageDir = OutTrainDir + "/Image/"
OutTrainLabelDir = OutTrainDir + "/Label/"


if not os.path.exists(OutEvalImageDir): os.makedirs(OutEvalImageDir)
if not os.path.exists(OutEvalLabelDir): os.makedirs(OutEvalLabelDir)

if not os.path.exists(OutTrainImageDir): os.makedirs(OutTrainImageDir)
if not os.path.exists(OutTrainLabelDir): os.makedirs(OutTrainLabelDir)
# ...............................create train folder.............................................................................
for TrainFn in Files:
    ClassName=TrainFn[:-10]
    with open(TxtFileDirs + "/" + TrainFn) as fp:
        for line in fp:
            FileName=line[0:line.find(" ")]
            if not "-1" in line and os.path.isfile(ImageDir+"/"+FileName+".jpg") and os.path.isfile(LabelDir + "/" + FileName + ".mat"):
               copyfile(LabelDir + "/" + FileName + ".mat", OutTrainLabelDir + "/" + FileName + ".mat")
               copyfile(ImageDir+"/"+FileName+".jpg",OutTrainImageDir +"/"+FileName+".jpg")
               print(FileName)

# ...............................create evaluation folder.............................................................................
    EvalFn = ClassName + "_val.txt"

    with open(TxtFileDirs + "/" + EvalFn) as fp:
        for line in fp:
            FileName = line[0:line.find(" ")]
            if not "-1" in line and os.path.isfile(ImageDir+"/"+FileName+".jpg") and os.path.isfile(LabelDir + "/" + FileName + ".mat"):
               copyfile(LabelDir + "/" + FileName + ".mat", OutEvalLabelDir + "/" + FileName + ".mat")
               copyfile(ImageDir+"/"+FileName+".jpg", OutEvalImageDir +"/"+FileName+".jpg")
               print(FileName)
