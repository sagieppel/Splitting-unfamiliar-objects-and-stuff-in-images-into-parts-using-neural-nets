# Evaluator net:
## See parent folder for usage instructions  
 The evaluator net is used to check and rank the generated segments. The ranking is done according to how well the input segment fits the best matching real segments in the image. The evaluator net is a simple convolutional net Resnet that receives three inputs: an image and a generated segment mask and object mask. The evaluator net predicts the intersection over union (IOU) between the input segment and the closest real segment in the image. In this case, the segment corresponds to the object part.


![](/Evaluator/Figure1.png)
## Requirements
This network was run with Python 3.7  [Anaconda](https://www.anaconda.com/download/) package and [Pytorch 1](https://pytorch.org/) and opencv. The training was done using Nvidia GTX 1080.

# Generating Training and Evaluation Data
The training and evaluation data is generated using the pointer net generator.
Hence it first necessary to train a pointer net and use it to generate predictions [PointerSegmentation]() for instruction on generating data. 
See the Example folder, for example, training/evaluation data.
# Training
1. Generate training data using the instructions above.
2. open Train.py
3. Set path to the generated a training data:

   AnnDir = Annotation folder
   
   ImageDir =  Image folder
   
4. Run script. The results will appear in the log folder.

# Evaluation
1. Train net or download pre-trained net [here] (or train the net)
2. Generate training data using the instructions above.
3. open Evaluate.py
4. Set path to the generated evaluation data:
   
   AnnDir = Annotation folder 
   
   ImageDir =  Image folder
   
5. Run script.
