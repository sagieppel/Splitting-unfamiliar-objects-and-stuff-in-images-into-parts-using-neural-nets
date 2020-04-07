
# Reader for part, read the part mask the object  mask (and object class) and an image
# The training data need to be prepared  by the pointer net (see script GenerateTrainingDataForEvaluator.py in pointer net for generation)
import numpy as np
import os
import scipy.misc as misc
import random
import cv2
import json
import threading
############################################################################################################
#########################################################################################################################
class Reader:
# Initiate reader and define the main parameters for the data reader
    def __init__(self, ImageDir,MaskDir,ClassBalance=True, MinPrecision=0.0,MaxBatchSize=100,MinSize=250,MaxSize=900,MaxPixels=800*800*5,TrainingMode=True,AugmentImage=False,Suffle=False, InverseSelection=False,UseAllClasses=False):
        self.MaxBatchSize = MaxBatchSize  # Max number of image in batch
        self.MinSize = MinSize  # Min image width and hight in pixels
        self.MaxSize = MaxSize  # Max image width and hight in pixels
        self.MaxPixels = MaxPixels  # Max number of pixel in all the batch (reduce to solve oom out of memory issues)
        self.Epoch = 0  # Training Epoch
        self.itr = 0  # Training iteratation
        self.ClassBalance = ClassBalance
        self.Findx = 0
        self.MinCatSize = 1
        self.AugmentImage=AugmentImage

        # ----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
        self.AnnotationList = []
        self.AnnotationByCat = {}

        # for i in range(NumClasses):
        #     self.AnnotationByCat.append([])

        # Ann_name.replace(".","__")+"##Class##"+str(seg["Class"])+"##IsThing##"+str(seg["IsThing"])+"IDNum"+str(ii)+".png"
        print("Creating file list for reader this might take a while")
#-----------------------Read list of classes mask and and images------------------------------------------------
        for uu, Name in enumerate(os.listdir(MaskDir)):
            if not ".png" in Name: continue
      #      if uu>100: break
            print(uu)
            s = {}
            s["MaskFile"] = MaskDir + "/" + Name
            s["ImageFile"] = ImageDir + "/" + Name[:-4] + ".jpg"
            s["IOU"] = float(Name[Name.find("#IOU#") + 5:Name.find("#Precision#")])
            s["Precision"] = float(Name[Name.find("#Precision#") + 11:Name.find("#Recall#")])
            s["Recall"] = float(Name[Name.find("#Recall#") + 8:Name.find('#Class#')])
            s["Class"] = int(Name[Name.find("#Class#") + 7:Name.find("#PartClass#")])
            s["PartClass"] = int(Name[Name.find("#PartClass#") + 11:Name.find("#PartLevel#")])
            s["PartLevel"] = int(Name[Name.find("#PartLevel#") + 11:Name.find("#RandID#")])


            if not (os.path.exists(s["ImageFile"]) and os.path.exists(s["MaskFile"])):
                print("Missing:" + s["MaskFile"])
                continue

            if s["PartLevel"] > 1: continue

            if s["Class"] not in self.AnnotationByCat:
                self.AnnotationByCat[s["Class"]] = []
            Select = (s["Class"] % 5) == 0  # ((s["Class"] % 7) == 0 or (s["Class"] % 9) == 0) # In case you want to keep unfamiliar/unseen class in training
   #         print(Select)
            if (not InverseSelection) and Select and (not UseAllClasses): continue
            if InverseSelection and not Select and (not UseAllClasses): continue
            self.AnnotationByCat[s["Class"]].append(s)
            self.AnnotationList.append(s)

        # tt=0
        # uu=0
        # for i,ct in enumerate(self.AnnotationByCat):
        #         print(str(i) + ")" + str(ct)+" "+str(len(self.AnnotationByCat[ct])))
        #         if (ct % 7) == 0  or (ct % 9) == 0 == 0:
        #             uu+=len(self.AnnotationByCat[ct])
        #             tt+=1
        #             self.AnnotationByCat[ct]=[]

        if Suffle:
            np.random.shuffle(self.AnnotationList)
        print("All cats " + str(len(self.AnnotationList)))
        print("done making file list")
        iii = 0
        if TrainingMode: self.StartLoadBatch()
#############################################################################################################################
# Crop and resize image and mask and ROI to feet batch size
    def CropResize(self,Img, Mask,AnnMap,Hb,Wb):
        # ========================resize image if it too small to the batch size==================================================================================
        bbox= cv2.boundingRect(Mask.astype(np.uint8))
        [h, w, d] = Img.shape
        Rs = np.max((Hb / h, Wb / w))
        Wbox = int(np.floor(bbox[2]))  # Segment Bounding box width
        Hbox = int(np.floor(bbox[3]))  # Segment Bounding box height
        if Wbox==0: Wbox+=1
        if Hbox == 0: Hbox += 1


        Bs = np.min((Hb / Hbox, Wb / Wbox))
        if Rs > 1 or Bs<1 or np.random.rand()<0.3:  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
            h = int(np.max((h * Rs, Hb)))
            w = int(np.max((w * Rs, Wb)))
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            Mask = cv2.resize(Mask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            AnnMap = cv2.resize(AnnMap.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            bbox = (np.float32(bbox) * Rs.astype(np.float)).astype(np.int64)

 # =======================Crop image to fit batch size===================================================================================
        x1 = int(np.floor(bbox[0]))  # Bounding box x position
        Wbox = int(np.floor(bbox[2]))  # Bounding box width
        y1 = int(np.floor(bbox[1]))  # Bounding box y position
        Hbox = int(np.floor(bbox[3]))  # Bounding box height

        if Wb > Wbox:
            Xmax = np.min((w - Wb, x1))
            Xmin = np.max((0, x1 - (Wb - Wbox)-1))
        else:
            Xmin = x1
            Xmax = np.min((w - Wb, x1 + (Wbox - Wb)+1))

        if Hb > Hbox:
            Ymax = np.min((h - Hb, y1))
            Ymin = np.max((0, y1 - (Hb - Hbox)-1))
        else:
            Ymin = y1
            Ymax = np.min((h - Hb, y1 + (Hbox - Hb)+1))

        if Ymax<=Ymin: y0=Ymin
        else: y0 = np.random.randint(low=Ymin, high=Ymax + 1)

        if Xmax<=Xmin: x0=Xmin
        else: x0 = np.random.randint(low=Xmin, high=Xmax + 1)

        # Img[:,:,1]*=Mask
        # misc.imshow(Img)

        Img = Img[y0:y0 + Hb, x0:x0 + Wb, :]
        Mask = Mask[y0:y0 + Hb, x0:x0 + Wb]
        AnnMap = AnnMap[y0:y0 + Hb, x0:x0 + Wb]
#------------------------------------------Verify shape match the batch shape----------------------------------------------------------------------------------------
        if not (Img.shape[0] == Hb and Img.shape[1] == Wb): Img = cv2.resize(Img, dsize=(Wb, Hb),interpolation=cv2.INTER_LINEAR)
        if not (Mask.shape[0] == Hb and Mask.shape[1] == Wb):Mask = cv2.resize(Mask.astype(float), dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)
        if not (AnnMap.shape[0] == Hb and AnnMap.shape[1] == Wb): AnnMap = cv2.resize(AnnMap.astype(float), dsize=(Wb, Hb),interpolation=cv2.INTER_NEAREST)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return Img,Mask,AnnMap
        # misc.imshow(Img)
#################################################Generate Annotaton mask#############################################################################################################333
#################################################Generate Pointer mask#############################################################################################################333
    def GeneratePointermask(self, Mask):
        bbox = cv2.boundingRect(Mask.astype(np.uint8))
        x1 = int(np.floor(bbox[0]))  # Bounding box x position
        Wbox = int(np.floor(bbox[2]))  # Bounding box width
        xmax = np.min([x1 + Wbox+1, Mask.shape[1]])
        y1 = int(np.floor(bbox[1]))  # Bounding box y position
        Hbox = int(np.floor(bbox[3]))  # Bounding box height
        ymax = np.min([y1 + Hbox+1, Mask.shape[0]])
        PointerMask=np.zeros(Mask.shape,dtype=np.float)
        if Mask.max()==0:return PointerMask

        while(True):
            x =np.random.randint(x1,xmax)
            y = np.random.randint(y1, ymax)
            if Mask[y,x]>0:
                PointerMask[y,x]=1
                return(PointerMask)
######################################################Augmented  image and mask##################################################################################################################################
    def Augment(self,Img,Mask,AnnMap,prob):
        if np.random.rand()<0.5: # flip left right
            Img=np.fliplr(Img)
            Mask = np.fliplr(Mask)
            AnnMap = np.fliplr(AnnMap)

        if np.random.rand()< (prob/12): # flip up down
            Img=np.flipud(Img)
            Mask = np.flipud(Mask)
            AnnMap = np.flipud(AnnMap)
        #
        # if np.random.rand() < prob: # resize
        #     r=r2=(0.6 + np.random.rand() * 0.8)
        #     if np.random.rand() < prob*0.2:  #Strech
        #         r2=(0.65 + np.random.rand() * 0.7)
        #     h = int(Mask.shape[0] * r)
        #     w = int(Mask.shape[1] * r2)
        #     Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        #     Mask = cv2.resize(Mask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        #     AnnMap = cv2.resize(AnnMap.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)

        if np.random.rand() < prob:  # Dark light
            Img = Img * (0.5 + np.random.rand() * 0.7)
            Img[Img>255]=255

        if np.random.rand() < prob:  # GreyScale
            Gr=Img.mean(axis=2)
            r=np.random.rand()

            Img[:, :, 0] = Img[:, :, 0] * r + Gr * (1 - r)
            Img[:, :, 1] = Img[:, :, 1] * r + Gr * (1 - r)
            Img[:, :, 2] = Img[:, :, 2] * r + Gr * (1 - r)


        return Img,Mask,AnnMap
########################################################################################################################################################

# ==========================Read image annotation and data===============================================================================================
    def LoadNext(self, batch_pos, Hb=-1, Wb=-1):
            if self.ClassBalance:  # pick with equal class probability
                while (True):
                    CL = random.choice(list(self.AnnotationByCat))
                    CatSize = len(self.AnnotationByCat[CL])
                    if CatSize >= self.MinCatSize: break
                Nim = np.random.randint(CatSize)
                # print("nim "+str(Nim)+"CL "+str(CL)+"  length"+str(len(self.AnnotationByCat[CL])))
                Ann = self.AnnotationByCat[CL][Nim]
            else:  # Pick with equal image probabiliry
                Nim = np.random.randint(len(self.AnnotationList))
                Ann = self.AnnotationList[Nim]
                CatSize = 100000000

            Img = cv2.imread(Ann["ImageFile"])  # Load Image
            Img = Img[..., :: -1]
            if (Img.ndim == 2):  # If grayscale turn to rgb
                Img = np.expand_dims(Img, 3)
                Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
            # -------------------------Read annotation--------------------------------------------------------------------------------
            Mask = cv2.imread(Ann["MaskFile"])  # Load mask
            ObjectMask = (Mask[:, :, 2] > 0).astype(float) # Object mask 
            Mask = (Mask[:, :, 0] > 0).astype(float) # Part mask

            # # -------------------------Augment-----------------------------------------------------------------------------------------------
            # if np.random.rand() < 0.62:
            #     Img, Mask, ObjectMask = self.Augment(Img, Mask, ObjectMask, np.min([float(200 / CatSize) * 0.29 + 0.01, 0.8]))
            # # -----------------------------------Crop and resize-----------------------------------------------------------------------------------------------------
            # if not Hb == -1:
            #     Img, Mask, ObjectMask = self.CropResize(Img, Mask, ObjectMask, Hb, Wb)

#----------------------------------------------------------------------------------------------------------------------------
            if not (Img.shape[0] == Mask.shape[0] and Img.shape[1] == Mask.shape[1]):
                if np.random.rand() < 0.8:
                    Mask = cv2.resize(Mask, (Img.shape[1], Img.shape[0]), interpolation=cv2.INTER_NEAREST)

                else:
                    #print("ResImage")
                    Img = cv2.resize(Img, (Mask.shape[1], Mask.shape[0]), interpolation=cv2.INTER_LINEAR)
#-----------------------------------------------------------------------------------------------------------------------------
            if (Mask is None) or  (Img is None):

                print("Missing "+ Ann["MaskFile"])
                self.LoadNext(batch_pos, Hb, Wb)
            else:
#-------------------------Augment-----------------------------------------------------------------------------------------------
                if self.AugmentImage:  Img,Mask,ObjectMask=self.Augment(Img,Mask,ObjectMask,np.min([float(200/CatSize)*0.29+0.01,0.8]))
#-----------------------------------------------------------------------------------------------------------------------------------------
                self.LabelFileName = Ann["MaskFile"]
# -----------------------------------Crop and resize-----------------------------------------------------------------------------------------------------
                if not Hb == -1:
                    # print(Img.shape)
                    # print(MaskGT.shape)
                    # print(MaskPred.shape)
                    #Img, Mask,ObjectMask = self.CropResize(Img, Mask, ObjectMask, Hb, Wb)
                    Img = cv2.resize(Img, (Wb, Hb), interpolation=cv2.INTER_LINEAR)
                    Mask = cv2.resize(Mask, (Wb, Hb), interpolation=cv2.INTER_NEAREST)
                    ObjectMask = cv2.resize(ObjectMask, (Wb, Hb), interpolation=cv2.INTER_NEAREST)

            #     Img, MaskGT, MaskPred = self.CropResize(Img2, MaskGT2, MaskPred2, Hb, Wb)
# ---------------------------------------------------------------------------------------------------------------------------------
                self.BImgs[batch_pos] = Img
                self.BPartMask[batch_pos] = Mask
                self.BIOU[batch_pos] = Ann["IOU"]
                self.BObjectMask[batch_pos] =  ObjectMask

                if Mask.max() == 0:
                    self.LoadNext(batch_pos,Hb, Wb)
############################################################################################################################################################
############################################################################################################################################################
# Start load batch of images, segment masks, and data
    def StartLoadBatch(self):
        # =====================Initiate batch=============================================================================================
        while True:
            Hb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width
            if Hb*Wb<self.MaxPixels: break
        BatchSize =  np.int(np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize)))
        self.BImgs = np.zeros((BatchSize, Hb, Wb, 3))  #
        self.BPartMask = np.zeros((BatchSize, Hb, Wb))
        self.BObjectMask = np.zeros((BatchSize, Hb, Wb))
        self.BIOU = np.zeros((BatchSize))
        #====================Start reading data multithreaded===========================================================
        self.thread_list = []
        for pos in range(BatchSize):
            th=threading.Thread(target=self.LoadNext,name="thread"+str(pos),args=(pos,Hb,Wb))
            self.thread_list.append(th)
            th.start()
        self.itr+=BatchSize
 ##################################################################################################################
    def SuffleFileList(self):
            random.shuffle(self.FileList)
            self.itr = 0
###########################################################################################################
#Wait until the data batch loading started at StartLoadBatch is finished
    def WaitLoadBatch(self):
            for th in self.thread_list:
                 th.join()

########################################################################################################################################################################################
    def LoadBatch(self):
# Load batch for training (muti threaded  run in parallel with the training proccess)
# For training
            self.WaitLoadBatch()
            Imgs=self.BImgs
            Mask=self.BPartMask
            IOU=self.BIOU
            ObjectMask=self.BObjectMask
            self.StartLoadBatch()
            return Imgs, Mask,ObjectMask,IOU
########################################################################################################################################################################################
#Load single with no augmentation
    def LoadSingle(self):
            # -----------------------------------Image and resize-----------------------------------------------------------------------------------------------------
            Finished=False
            Ann = self.AnnotationList[self.Findx ]
            self.Findx +=1
            if self.Findx>=len(self.AnnotationList):
                self.Findx=0
                self.Epoch+=1
                Finished=True

            #  print("Nor")
           # print(Ann["GTMaskFile"])
            Img = cv2.imread(Ann["ImageFile"])  # Load Image
            Img = Img[..., :: -1]
            if (Img.ndim == 2):  # If grayscale turn to rgb
                Img = np.expand_dims(Img, 3)
                Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
            # -------------------------Read annotation--------------------------------------------------------------------------------
            Mask = cv2.imread(Ann["MaskFile"])  # Load mask
            ObjectMask = (Mask[:, :, 2] > 0).astype(float)
            Mask = (Mask[:, :, 0] > 0).astype(float)
            Cat=Ann["Class"]

            #-------------------------------------------------------------------------------------------------------------------
            Mask=np.expand_dims(Mask,axis=0).astype(np.float32)
            Img = np.expand_dims(Img, axis=0).astype(np.float32)
            ObjectMask= np.expand_dims(ObjectMask, axis=0).astype(np.float32)
            return Img,Mask, ObjectMask,Ann["IOU"],Cat,Finished

