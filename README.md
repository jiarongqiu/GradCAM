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

```
    # TODO create your own process
    def preprocess(self, x):
        x = x / 255.0
        x = x - np.array([0.485, 0.456, 0.406])
        x = x / np.array([0.229, 0.224, 0.225])
        return x

    # TODO fetch key tensors
    def fetch_key_tensors(self):
        """
            X:input tensor
            logit:logit output tensor
            layer:gradient dependent layer to the logit
            ... :fetch and set other tensors and their default values for inference
        """
        self.X = self.sess.graph.get_tensor_by_name('input:0')
        self.logit = self.sess.graph.get_tensor_by_name("finetune_dense1/BiasAdd:0")
        self.layer = self.sess.graph.get_tensor_by_name("ResNetnSequentialnlayer4nnBasicBlockn1nnReLUnrelun168:0")

        is_training = self.sess.graph.get_tensor_by_name("Placeholder:0")
        self.feed_dict = {is_training:False}
```

After that, simply use the following function to added saliency map to the images
```
    visualizer =GradCam(model_dir,nb_classes)
    visualizer.visualize_imgs(imgs_path,out_folder)
```


Please reference the following for further details:

[model explanation](https://bindog.github.io/blog/2018/02/10/model-explanation/)

[grad-cam.tensorflow](https://github.com/Ankush96/grad-cam.tensorflow)