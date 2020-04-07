
#Reader for the for annotation for object parts (data need to be generated using  the script at GenerateTrainingData
import numpy as np
import os
import scipy.misc as misc
import random
import cv2
import json
import threading
import random
############################################################################################################
#########################################################################################################################
class Reader:
# Initiate reader and define the main parameters for the data reader
    def __init__(self, ImageDir,AnnDir,ClassBalance=True, MaxBatchSize=100,MinSize=250,MaxSize=1000,MaxPixels=800*800*5, AnnotationFileType="png", ImageFileType="jpg",TrainingMode=True,Suffle=False,  InverseSelection=False, UseAllClaseses=False):

        self.MaxBatchSize=MaxBatchSize # Max number of image in batch
        self.MinSize=MinSize # Min image width and hight in pixels
        self.MaxSize=MaxSize #Max image width and hight in pixels
        self.MaxPixels=MaxPixels # Max number of pixel in all the batch (reduce to solve oom out of memory issues)
        self.AnnotationFileType=AnnotationFileType # What is the the type (ending) of the annotation files
        self.ImageFileType=ImageFileType # What is the the type (ending) of the image files
        self.Epoch = 0 # Training Epoch
        self.itr = 0 # Training iteratation
        self.ClassBalance=ClassBalance
        self.Findx=0
        self.MinCatSize=1
# ----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
        self.AnnotationList = []
        self.AnnotationByCat = {}


        # for i in range(NumClasses):
        #     self.AnnotationByCat.append([])

        #Ann_name.replace(".","__")+"##Class##"+str(seg["Class"])+"##IsThing##"+str(seg["IsThing"])+"IDNum"+str(ii)+".png"
        print("Creating file list for reader this might take a while")

        for uu,Name in enumerate(os.listdir(AnnDir)):
                #####if uu>1000: break
           #     print(uu)
                s = {}
                s["MaskFile"] = AnnDir + "/" + Name
                s["ImageFile"] = ImageDir+"/"+Name[:Name.find("_PartsLevel_")]+".jpg"
                s["PartLevel"]=int(Name[Name.find("_PartsLevel_")+12:Name.find("_PartInstance_")])
                s["PartClass"]=int(Name[Name.find("_PartClass_")+11:Name.find("_TopClass_")])
                s["Class"] = int(Name[Name.find("_TopClass_")+10:Name.find(".png")])
                if not (os.path.exists(s["ImageFile"]) and os.path.exists(s["MaskFile"])):
                                      print("Missing:"+s["MaskFile"])
                                      continue


                if s["PartLevel"]>1: continue

                if s["Class"] not in self.AnnotationByCat:
                                self.AnnotationByCat[s["Class"]]=[]
                Select = (s["Class"] % 5) == 0 #((s["Class"] % 7) == 0 or (s["Class"] % 9) == 0) # In case you want to keep unfamiliar/unseen class in training
                #         print(Select)
                if (not InverseSelection) and Select and (not UseAllClaseses): continue
                if InverseSelection and not Select and (not UseAllClaseses) : continue
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
        print("All cats "+str(len(self.AnnotationList)))
        print("done making file list")
        iii=0
        if TrainingMode: self.StartLoadBatch()
        self.AnnData=False
#############################################################################################################################
# Crop and resize image and mask and Object mask to feet batch size
    def CropResize(self,Img, PartMask,AnnMap,Hb,Wb):
        # ========================resize image if it too small to the batch size==================================================================================
        bbox= cv2.boundingRect(PartMask.astype(np.uint8))
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
            PartMask = cv2.resize(PartMask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
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

        # Img[:,:,1]*=PartMask
        # misc.imshow(Img)

        Img = Img[y0:y0 + Hb, x0:x0 + Wb, :]
        PartMask = PartMask[y0:y0 + Hb, x0:x0 + Wb]
        AnnMap = AnnMap[y0:y0 + Hb, x0:x0 + Wb]
#------------------------------------------Verify shape match the batch shape----------------------------------------------------------------------------------------
        if not (Img.shape[0] == Hb and Img.shape[1] == Wb): Img = cv2.resize(Img, dsize=(Wb, Hb),interpolation=cv2.INTER_LINEAR)
        if not (PartMask.shape[0] == Hb and PartMask.shape[1] == Wb):PartMask = cv2.resize(PartMask.astype(float), dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)
        if not (AnnMap.shape[0] == Hb and AnnMap.shape[1] == Wb): AnnMap = cv2.resize(AnnMap.astype(float), dsize=(Wb, Hb),interpolation=cv2.INTER_NEAREST)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return Img,PartMask,AnnMap
        # misc.imshow(Img)
#################################################Generate Annotaton mask#############################################################################################################333
#################################################Generate Pointer mask#############################################################################################################333
    def GeneratePointermask(self, PartMask):
        bbox = cv2.boundingRect(PartMask.astype(np.uint8))
        x1 = int(np.floor(bbox[0]))  # Bounding box x position
        Wbox = int(np.floor(bbox[2]))  # Bounding box width
        xmax = np.min([x1 + Wbox+1, PartMask.shape[1]])
        y1 = int(np.floor(bbox[1]))  # Bounding box y position
        Hbox = int(np.floor(bbox[3]))  # Bounding box height
        ymax = np.min([y1 + Hbox+1, PartMask.shape[0]])
        PointerMask=np.zeros(PartMask.shape,dtype=np.float)
        if PartMask.max()==0:return PointerMask

        while(True):
            x =np.random.randint(x1,xmax)
            y = np.random.randint(y1, ymax)
            if PartMask[y,x]>0:
                PointerMask[y,x]=1
                return(PointerMask)
######################################################Augmented mask##################################################################################################################################
    def Augment(self,Img,PartMask,AnnMap,prob):
        if np.random.rand()<0.5: # flip left right
            Img=np.fliplr(Img)
            PartMask = np.fliplr(PartMask)
            AnnMap = np.fliplr(AnnMap)

        if np.random.rand()< (prob/12): # flip up down
            Img=np.flipud(Img)
            PartMask = np.flipud(PartMask)
            AnnMap = np.flipud(AnnMap)
        #
        # if np.random.rand() < prob: # resize
        #     r=r2=(0.6 + np.random.rand() * 0.8)
        #     if np.random.rand() < prob*0.2:  #Strech
        #         r2=(0.65 + np.random.rand() * 0.7)
        #     h = int(PartMask.shape[0] * r)
        #     w = int(PartMask.shape[1] * r2)
        #     Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        #     PartMask = cv2.resize(PartMask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
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


        return Img,PartMask,AnnMap
########################################################################################################################################################
# ==========================Read image annotation and data===============================================================================================
    def LoadNext(self, batch_pos, Hb=-1, Wb=-1):
            if self.ClassBalance: # pick with equal class probability
                while (True):
                     CL=random.choice(list(self.AnnotationByCat))
                     if not (CL in self.AnnotationByCat): continue
                     CatSize=len(self.AnnotationByCat[CL])
                     if CatSize>=self.MinCatSize: break
                Nim = np.random.randint(CatSize)
               # print("nim "+str(Nim)+"CL "+str(CL)+"  length"+str(len(self.AnnotationByCat[CL])))
                Ann=self.AnnotationByCat[CL][Nim]
            else: # Pick with equal probability per annotation
                Nim = np.random.randint(len(self.AnnotationList))
                Ann=self.AnnotationList[Nim]
                CatSize=100000000
#--------------Read image--------------------------------------------------------------------------------
            Img = cv2.imread(Ann["ImageFile"])  # Load Image
            Img = Img[..., :: -1]
            if (Img.ndim == 2):  # If grayscale turn to rgb
                Img = np.expand_dims(Img, 3)
                Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
#-------------------------Read annotation--------------------------------------------------------------------------------
            PartMask = cv2.imread(Ann["MaskFile"])  # Load mask
            ObjecMask= (PartMask[:,:,1]>0).astype(float) # Object mask
            PartMask= (PartMask[:,:,0]>0).astype(float)

#-------------------------Augment-----------------------------------------------------------------------------------------------
            if np.random.rand()<0.62:
                          Img,PartMask,ObjecMask=self.Augment(Img,PartMask,ObjecMask,np.min([float(200/CatSize)*0.29+0.01,0.8]))
#-----------------------------------Crop and resize-----------------------------------------------------------------------------------------------------
            if not Hb==-1:
               Img, PartMask,ObjecMask = self.CropResize(Img, PartMask, ObjecMask, Hb, Wb)
#----------------------------------------------------------------------------------------------------------------------------------
            PointerMask=self.GeneratePointermask(PartMask)
#---------------------------------------------------------------------------------------------------------------------------------
            self.BPointerMask[batch_pos] =  PointerMask
            self.BROIMask[batch_pos] =  ObjecMask
            self.BImgs[batch_pos] = Img
            self.BSegmentMask[batch_pos] = PartMask
            self.BCatID[batch_pos] = Ann["Class"]

############################################################################################################################################################
# Start load batch of images, segment masks, ROI masks, and pointer points for training MultiThreading s
    def StartLoadBatch(self):
        # =====================Initiate batch=============================================================================================
        while True:
            Hb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width
            if Hb*Wb<self.MaxPixels: break
        BatchSize =  np.int(np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize)))
        self.BImgs = np.zeros((BatchSize, Hb, Wb, 3))  #
        self.BSegmentMask = np.zeros((BatchSize, Hb, Wb))
        self.BROIMask = np.zeros((BatchSize, Hb, Wb))
        self.BPointerMask = np.zeros((BatchSize, Hb, Wb))
        self.BCatID = np.zeros((BatchSize))
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
            SegmentMask=self.BSegmentMask
            CatID=self.BCatID
            ObjecMask = self.BROIMask
            PointerMask = self.BPointerMask
            self.StartLoadBatch()
            return Imgs, SegmentMask,ObjecMask,PointerMask
#Imgs, SegmentMask, ObjecMask, PointerMap
############################Load single data with no augmentation############################################################################################################################################################
    def LoadSingle(self):
            # ---------------------------------------------------------------------------------------------------------------
            #print("findx " + str(self.Findx))
            if self.Findx >= len(self.AnnotationList):
                self.Findx = int(0)
                self.Epoch += 1
            Ann = self.AnnotationList[self.Findx]
            self.Findx += 1

            # -------------------------image--------------------------------------------------------------------------------
            Img = cv2.imread(Ann["ImageFile"])  # Load Image
            Img = Img[..., :: -1]
            if (Img.ndim == 2):  # If grayscale turn to rgb
                Img = np.expand_dims(Img, 3)
                Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
            # -------------------------Read annotation--------------------------------------------------------------------------------
            PartMask = cv2.imread(Ann["MaskFile"])  # Load mask
            FullAnn = PartMask[:, :, 2]
            ObjecMask = (PartMask[:, :, 1] > 0).astype(float)
            PartMask = (PartMask[:, :, 0] > 0).astype(float)

            PointerMask = self.GeneratePointermask(PartMask)
            PointerMask = np.expand_dims(PointerMask, axis=0).astype(np.float)
            PartMask = np.expand_dims(PartMask, axis=0).astype(np.float)
            Img = np.expand_dims(Img, axis=0).astype(np.float)
            ObjecMask = np.expand_dims(ObjecMask, axis=0).astype(np.float)
            fname = Ann["ImageFile"]
            return Img, PartMask, PointerMask, ObjecMask, FullAnn, fname[fname.rfind("/") + 1:], Ann["PartLevel"], Ann["PartClass"],Ann["Class"]

########################################################################################################################################################################################
####################################################################################################################################
    def LoadSingleForGeneration(self,Augment=True):
        while True:
            Hb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width
            if Hb*Wb<self.MaxPixels: break
        #---------------------------------------------------------------------------------------------------------------
        print("findx " + str(self.Findx))
        if self.Findx>=len(self.AnnotationList):
            self.Findx=int(0)
            self.Epoch+=1
        Ann = self.AnnotationList[self.Findx]




        # -------------------------image--------------------------------------------------------------------------------
        Img = cv2.imread(Ann["ImageFile"])  # Load Image
        Img = Img[..., :: -1]
        if (Img.ndim == 2):  # If grayscale turn to rgb
            Img = np.expand_dims(Img, 3)
            Img = np.concatenate([Img, Img, Img], axis=2)
        Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
        # -------------------------Read annotation--------------------------------------------------------------------------------
        PartMask = cv2.imread(Ann["MaskFile"])  # Load mask
        ObjecMask = (PartMask[:, :, 1] > 0).astype(float)
        PartMask = (PartMask[:, :, 0] > 0).astype(float)


        # -------------------------Augment-----------------------------------------------------------------------------------------------
        if np.random.rand() < 0.9:
            Img, PartMask, ObjecMask = self.Augment(Img, PartMask, ObjecMask,0.5)
#---------------------------------------------------------------------------------------------------------
        Img, PartMask, ObjecMask = self.CropResize(Img, PartMask, ObjecMask, Hb, Wb)
        #------------------------------------------------------------------------------------------------------------------------------------

        PointerMask = self.GeneratePointermask(PartMask)
        PointerMask = np.expand_dims(PointerMask, axis=0).astype(np.float)
        PartMask = np.expand_dims(PartMask, axis=0).astype(np.float)
        Img = np.expand_dims(Img, axis=0).astype(np.float)
        ObjecMask = np.expand_dims(ObjecMask, axis=0).astype(np.float)
        fname=Ann["ImageFile"]
        self.Findx += 1
        return Img, PartMask,PointerMask,ObjecMask, fname[fname.rfind("/")+1:],Ann["PartLevel"],Ann["PartClass"],Ann["Class"]
############################################################################################################################################################
    def Reset(self):
        self.Cindx = int(0)
        self.Findx = int(0)
        self.CindList = np.zeros([len(self.AnnotationByCat)], dtype=int)
        self.Clepoch = np.zeros([len(self.AnnotationByCat)], dtype=int)
        self.Epoch = int(0)#not valid or

########################################








