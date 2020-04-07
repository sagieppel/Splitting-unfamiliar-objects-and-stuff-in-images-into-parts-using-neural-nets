import ReaderParts as  ReaderADE_Parts
import scipy.misc as misc
import numpy as np
MaxBatchSize=1 # Max images in batch
MinSize=250 # Min image Height/Width
MaxSize=1000# Max image Height/Width
MaxPixels=340000*6#4# Max pixel in batch can have (to keep oom out of memory problems) if the image larger it will be resized.
AnnDir="../TrainingDataForEvaluator/Ann/"
ImageDir="../TrainingDataForEvaluator/Img/"
Reader = ReaderADE_Parts.Reader(ImageDir=ImageDir,MaskDir=AnnDir, MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,TrainingMode=True)
while(True):
    Reader.ClassBalance=False
    Reader.MinCatSize=np.random.randint(100)+1
    Imgs, Mask,ROIMask,IOU = Reader.LoadBatch()
    #Imgs, SegmentMask, PointerMap, ROIMask, fname, PartLevel, PartClass, Class = Reader.LoadSingle()
    for f in range(Imgs.shape[0]):
            print(IOU[f])
            Imgs[f, :, :, 0] *= 1-Mask[f]
            Imgs[f, :, :, 1] *= ROIMask[f]
            misc.imshow(Imgs[f])
            misc.imshow((ROIMask[f] + Mask[f] * 3).astype(np.uint8)*40)
            print("ee")