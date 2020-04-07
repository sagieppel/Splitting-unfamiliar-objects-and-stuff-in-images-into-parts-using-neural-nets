# Convert ADE20k files to training data for the net

import cv2
import numpy as np
import os
import scipy.misc as misc
###################Input parameters#########################################################
Ade20kMainDir =r"/media/sagi/2T/Data_zoo/ADE20K_2016_07_26/images/training/" # Input dir
OutLabelDir=r"/media/sagi/2T/Data_zoo/ADE20K_2016_07_26/Label/" # ouput image dir
OutImageDir=r"/media/sagi/2T/Data_zoo/ADE20K_2016_07_26/Image/"# output annotation dir
####################################################################################################
if not os.path.exists(Ade20kMainDir ): print("input dir emptr")
if not os.path.exists(OutLabelDir): os.mkdir(OutLabelDir)
if not os.path.exists(OutImageDir): os.mkdir(OutImageDir)

#####################visualization###################################3
def show(img,txt=""):
    misc.imshow(img.astype(np.uint8))
   # cv2.imshow(txt, img.astype(np.uint8))  # Display on screen
    kboard = chr(cv2.waitKey())  # wait until key is pressed or 5 seconds have passed
    return kboard

##################################Find dominant index in a segment##################333
def domindx(im,mask):
    mk=im * mask
    ind=0
    mxsm=0
    for i in np.unique(mk):
        if i>0:
           sm=(mk==i).sum()
           if sm>mxsm:
               mxsm=sm
               ind=i
    return(ind)

###################################################

for r,d,f in os.walk(Ade20kMainDir ):
         for fl in f:
          #  print(r+"/"+fl)
            if "_parts_" in fl:
                ImPart=cv2.imread(r+"/"+fl)
                name=fl[:fl.find("_parts_")]
                ImgPath=r+"/"+name+".jpg"
                partnum=fl[fl.find("_parts_")+7:fl.find("_parts_")+8]
                if partnum=="1":
                    TopPath=r+"/"+name+"_seg.png"
                else:
                    TopPath = r + "/" + name + "_parts_"+str(int(partnum)-1)+".png"

                ImPart = cv2.imread(r + "/" + fl)
                ImTop = cv2.imread(TopPath)
                Img = cv2.imread(ImgPath)

                # show(cv2.resize(np.concatenate([Img,ImTop,ImPart],axis=1),(1500,500)))  # Display on screen
                # show((Img+ImPart)/2)  # Display on screen
                cv2.imwrite(OutImageDir + "/" + name + ".jpg", Img)
                print(fl)

     #..................Create map of all segments in annotation........................................................
                AllParts=np.zeros(ImPart[:, :, 0].shape,np.uint8)
                f=0
                for i in np.unique(ImPart[:, :, 0]):
                       if i==0: continue
                       f+=1
                       AllParts[ImPart[:, :, 0]==i]=f
     #..................Save each segment seperately........................................................
                for i in np.unique(ImPart[:,:,0]):
                         if i>0:
                             #----------Get Masks-------------------------------
                             PartMask=(ImPart[:,:,0]==i)

                             TopIns=domindx(ImTop[:,:,0],PartMask)
                             TopMask=(ImTop[:,:,0]==int(TopIns)) # Object mask
                             #-----------Get Class------------------------------------------
                             if TopMask.sum()==0 or PartMask.sum()==0:
                                 print("nan mask")
                                 continue
                             R=domindx(ImTop[:,:,2],TopMask)
                             G=domindx(ImTop[:,:,1],TopMask)
                             TopClass= (int(R) / 10) * 256 + int(G)

                             R = domindx(ImPart[:, :, 2] , PartMask)
                             G = domindx(ImPart[:, :, 1] , PartMask)
                             PartClass = (int(R) / 10) * 256 + int(G)
                             #---------save to file-----------------------------------------
                             outLabelpath= OutLabelDir + "/" + name + "_PartsLevel_"+str(int(partnum))+str("_PartInstance_"+str(int(i))+"_PartClass_"+str(int(PartClass))+"_TopClass_"+str(int(TopClass))+".png")
                             Label=np.concatenate([np.expand_dims(PartMask*255,axis=2),np.expand_dims(TopMask*255,axis=2),np.expand_dims((AllParts*TopMask).astype(np.uint8),axis=2)],axis=2)
                             cv2.imwrite(outLabelpath, Label)

                             #print(outLabelpath)
                             # show(Label)

