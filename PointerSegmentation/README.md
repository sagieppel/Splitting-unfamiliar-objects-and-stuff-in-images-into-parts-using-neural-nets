# Split  Unfamiliar object and stuff region in image into parts using Pointer net:
## See parent folder for usage instructions  
The system receives an image and object region in the image, and split/segment the object into parts. This is class agnostic and works for objects and non-objects (stuff). 
The training was done on the ADE20k and Pascal Parts data set.

See this [https://arxiv.org/pdf/1908.09108.pdf](https://arxiv.org/pdf/1908.09108.pdf) for more details on the method 
# Pointer net
Pointer net is a fully convolutional net that receives an image, a mask, of an object region and a point inside the object. The net predicts the region of the part that contains the input point (Figure 1). The generated output segment region will be confined to the Object mask. Selecting random points inside the object and then merging the prediction into one annotation map allows for full image segmentation.
Fully trained system ready to run could be found [here](IIIIIIIIIIIIIII)
The net runs best as part of the generator evaluator selector (GES) approach. See the parent folder for more details.



![](/PointerSegmentation/Figure1.png)
Figure 1) Pointer part segmentation architecture 

![](/PointerSegmentation/Figure2.png)
Figure 2) Using pointer segmentation for full object into part segmentation
## Requirements
This network was run with Python 3.7  [Anaconda](https://www.anaconda.com/download/) package and [Pytorch 1](https://pytorch.org/) and opencv. The training was done using Nvidia GTX 1080.

# Running the net:
1. Download the pre-trained system from [here](IIIIIIIIIIIIIII) or train using the instruction in training
2. Go to RunSegmentation.py and run (to run on example data).

## Parameters:
Trained_model_path = path to trained model
InImage = path input image
InObjectMask = path object mask region as a binary mask
OutAnnotationFile = Output annotation file path
See Example inputs for example inputs





# Generating Data for training and evaluation
## Generate training data using ADE20k (recommend)
1. Download ADE20k from [here](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
2. Open ConvertADE20kToTraining.py script inside the GenerateTrainingData folder
3. Set input and output paths in the input parameters section:

 Ade20kMainDir = Path to the ADE20k main folder   (ADE20K_2016_07_26/images/training/")

OutLabelDir = folder where the  converted output labels will be saved 

OutImageDir= folder where images will be saved

See Example folder for example training/evaluation data.

4. Run Script

## Generate training data using Pascal Parts
1. Download pascal parts  annotation from [here](http://roozbehm.info/pascal-parts/pascal-parts.html) and [images](https://cs.stanford.edu/~roozbeh/pascal-context/)
2. Open ConvertPascalPartsToTraining.py script inside the GenerateTrainingData folder
3. Set input and output paths in the input parameters section:

PascalPartLabelDir = Annotation Folder from the pascal part 

PascalImageDir = Image folder from the pascal dataset

OutDir = Folder where converted annotations will be saved

4. Run Script. 

# Training
1. Generate training data using the instructions above
2. Open TRAIN.py
3. Set paths to the training data in the input data section:

AnnDir = Path to the generated training annotations

ImageDir = Path to the generated image dir

4. Run script. The trained model will be saved into the log folder.

# Generating Training data for the evaluator net
If you using this system as part of the Generator evaluator system you can use the trained model to generate training data for the evaluator net.
1. Open GenerateTrainingDataForEvaluator.py script
2. Update input parameters in the Input parameters section:

modelDir = folder where the trained models are stored ( logs/)

AnnDir = Path to the generated training annotations folder

ImageDir = Path to the generated image folder 

OutDir = Path to the generated  evaluator training data will be saved

