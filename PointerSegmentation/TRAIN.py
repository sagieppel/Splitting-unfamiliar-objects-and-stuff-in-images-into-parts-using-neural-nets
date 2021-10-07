# Train pointer net for segmentation of parts in object. Net receiv object mask and point in the object and output the part mask
# Training data need to be generated using  the script at GenerateTrainingData
#...............................Imports..................................................................
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
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#.................................Main Input parametrs...........................................................................................
#AnnDir="/media/sagi/2T/GES_PARTS/PascalPartData/Train/Label/"
#ImageDir="/media/sagi/2T/GES_PARTS/PascalPartData/Train/Image//"
#
AnnDir="Example/Training/Anns/"
ImageDir="Example/Training/Images//"

#**************************************Train paprameter**************************************************************************************************************
TrainedModelWeightDir="logs/" # Folder where trained model weight and information will be stored"
if not os.path.exists(TrainedModelWeightDir): os.mkdir(TrainedModelWeightDir)
Trained_model_path="" # Path of trained model weights If you want to return to trained model, else should be =""
Learning_Rate_Init=1e-4 # Initial learning rate
Learning_Rate=1e-5 # learning rate
#=========================Load Paramters====================================================================================================================
InitStep=1
if os.path.exists(TrainedModelWeightDir + "/Defult.torch"):
    Trained_model_path=TrainedModelWeightDir + "/Defult.torch"
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate.npy"):
    Learning_Rate=np.load(TrainedModelWeightDir+"/Learning_Rate.npy")
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate_Init.npy"):
    Learning_Rate_Init=np.load(TrainedModelWeightDir+"/Learning_Rate_Init.npy")
if os.path.exists(TrainedModelWeightDir+"/itr.npy"): InitStep=int(np.load(TrainedModelWeightDir+"/itr.npy"))
#...............Other training paramters..............................................................................


MaxBatchSize=7 # Max images in batch
MinSize=250 # Min image Height/Width
MaxSize=1000# Max image Height/Width


#Learning_Rate_Decay=Learning_Rate[0]/40 # Used for standart
Learning_Rate_Decay=Learning_Rate/20
StartLRDecayAfterSteps=50000
MaxPixels=340000*3#4# Max pixel in batch can have (to keep oom out of memory problems) if the image larger it will be resized.
TrainLossTxtFile=TrainedModelWeightDir+"TrainLoss.txt" #Where train losses will be writen
Weight_Decay=1e-5# Weight for the weight decay loss function
MAX_ITERATION = int(100000010) # Max  number of training iteration
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained
Net=Net.cuda()
if Trained_model_path!="": # Optional initiate full net by loading a pretrained net
    Net.load_state_dict(torch.load(Trained_model_path))
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay) # Create adam optimizer
torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + "test" + ".torch")
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader=Data_Reader.Reader(ImageDir,AnnDir, MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,TrainingMode=True)

#optimizer=torch.optim.SGD(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay,momentum=0.5)

#--------------------------- Create logs files for saving loss during training----------------------------------------------------------------------------------------------------------
if not os.path.exists(TrainedModelWeightDir): os.makedirs(TrainedModelWeightDir) # Create folder for trained weight
f = open(TrainLossTxtFile, "w+")# Training loss log file
f.write("Iteration\tloss\t Learning Rate=")
f.close()
#..............Start Training loop: Main Training....................................................................
AVGLoss=-1# running average loss
print("Start Training")
for itr in range(InitStep,MAX_ITERATION): # Main training loop
   # print(itr)
    Reader.ClassBalance = np.random.rand() < 0.6
    Imgs, PartMask, ObjectMask, PointerMap = Reader.LoadBatch()

    # for f in range(Imgs.shape[0]):
    #     Imgs[f, :, :, 0] *= 1-PartMask[f]
    #     Imgs[f, :, :, 1] *= ObjectMask[f]
    #     misc.imshow(Imgs[f])
    #     misc.imshow((ObjectMask[f] + PartMask[f] * 2 + PointerMap[f] * 3).astype(np.uint8)*40)
    # print(ObjectMask.shape)
    # for i in range(1):  # Imgs.shape[0]):
    #       print(Imgs.shape)
    #       Reader.DisplayTrainExample(Imgs[i], ObjectMask[i], PartMask[i], PointerMap[i])

    OneHotLabels = ConvertLabelToOneHotEncoding.LabelConvert(PartMask, 2) #Convert labels map to one hot encoding pytorch
    #print("RUN PREDICITION")
    Prob, Lb=Net.forward(Images=Imgs,Pointer=PointerMap,ROI=ObjectMask) # Run net inference and get prediction
    Net.zero_grad()
    Loss = -torch.mean((OneHotLabels * torch.log(Prob + 0.0000001)))  # Calculate loss between prediction and ground truth label
    Loss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient descent change to weight
    if AVGLoss==-1:  AVGLoss=float(Loss.data.cpu().numpy()) #Calculate average loss for display
    else: AVGLoss=AVGLoss*0.999+0.001*float(Loss.data.cpu().numpy()) # Intiate runing average loss
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 2000 == 0:# and itr>0: #Save model weight once every 10k steps
        print("Saving Model to file in "+TrainedModelWeightDir+"/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/DefultBack.torch")
        print("model saved")
        np.save(TrainedModelWeightDir+"/Learning_Rate.npy",Learning_Rate)
        np.save(TrainedModelWeightDir+"/Learning_Rate_Init.npy",Learning_Rate_Init)
        np.save(TrainedModelWeightDir+"/itr.npy",itr)
    if itr % 40000 == 0 and itr>0: #Save model weight once every n steps
        print("Saving Model to file in "+TrainedModelWeightDir+"/"+ str(itr) + ".torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch")
        print("model saved")
        #Reader = Data_Reader.Reader(ImageDir, AnnDir, MaxBatchSize=MaxBatchSize, MinSize=MinSize, MaxSize=MaxSize,MaxPixels=MaxPixels, TrainingMode=True)
#......................Write and display train loss..........................................................................
    if itr % 10==0: # Display train loss

        txt="\n"+str(itr)+"\t Loss="+str(float(Loss.data))+"\t AverageLoss="+str(AVGLoss)+"\t Learning Rate="+str(Learning_Rate)+" Init_LR"+str(Learning_Rate_Init)
        print(txt)
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write(txt)
            f.close()
#----------------Update learning rate fractal manner-------------------------------------------------------------------------------
    if itr%10000==0 and itr>=StartLRDecayAfterSteps:
        Learning_Rate-= Learning_Rate_Decay
        if Learning_Rate<=1e-6:
            Learning_Rate_Init-=2e-6
            if Learning_Rate_Init<1e-6: Learning_Rate_Init=1e-6
            Learning_Rate=Learning_Rate_Init
            Learning_Rate_Decay=Learning_Rate/20
        print("Learning Rate="+str(Learning_Rate)+"   Learning_Rate_Init="+str(Learning_Rate_Init))
        print("======================================================================================================================")
        optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate,weight_decay=Weight_Decay)  # Create adam optimizer
        torch.cuda.empty_cache()  # Empty cuda memory to avoid memory leaks

