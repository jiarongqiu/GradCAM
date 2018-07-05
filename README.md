# GradCAM
TensorFlow Network layer attention visualizer

## GradCAM Principle 

GradCAM visualize the saliency map of feature maps to the prediction class. In details, it cals every feature map's weights ![](http://latex.codecogs.com/gif.latex?\\alpha_k^c) and its output ![](http://latex.codecogs.com/gif.latex?A^k). Then, apply RELU to filter out negative correlation points.

![](http://latex.codecogs.com/gif.latex?L_{Grad-CAM}=RELU(\\sum_k{\\alpha_k^cA^k}))

Then we cals its weights:

![](http://latex.codecogs.com/gif.latex?\\alpha_k^c=\\frac{1}{Z}\\sum_i\\sum_j\\frac{\\partial{y^c}}{\\partial{A^k_{ij}}})

where Z is the pixel nums of feature map.

## GradCAM Usage

Overide the following part for preprocessing and fetching the input tensor and auxiliary tensors 
