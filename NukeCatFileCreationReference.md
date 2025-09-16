Sourced from https://learn.foundry.com/nuke/developers/latest/catfilecreationreferenceguide/

Introduction
Welcome to the .cat File Creation guide. This guide details how to convert your PyTorch models to TorchScript, and from TorchScript to a .cat file that can be used in the Inference node.

We start this guide by presenting a simple object detection model that can be easily converted to a .cat file using the CatFileCreator node in Nuke. We then build on this simple example in the subsequent sections to highlight some of the things you should bear in mind when writing a model that is to be converted to a .cat file. These include:

ensuring your model forward function has input and output tensors that are the correct shape

ensuring you have added any necessary normalisation/preprocessing steps to your model forward function

ensuring your model works with Nuke’s Use GPU if available and Optimise for speed and Memory knobs

ensuring your model is properly annotated so that it can be successfully converted from PyTorch to TorchScript

adding attributes to your model that can be controlled using a custom knob in Inference

how to reformat your model so that it only has one input and output tensor, which is a requirement when running your model in Nuke

For these scenarios, we also highlight how to set up the CatFileCreator and Inference nodes so that you can create and use your .cat file successfully in Nuke. Finally, we present two examples at the end of this guide that combine some of the topics discussed.

Important

To create a .cat file that can be run in all versions of Nuke from Nuke 13.x onwards, the following libraries must be used when converting a PyTorch model to TorchScript:

torch==1.6 (https://pytorch.org)

torchvision==0.7

TorchScript models created using PyTorch 1.6 can be converted to .cat files using Nuke 13.x or Nuke 14. Note that TorchScript models intended to run in Nuke 13.x should be converted to .cat file using Nuke 13.x as those converted with Nuke 14 will not run in Nuke 13.x.

Nuke 14 now uses PyTorch 1.12 which means that PyTorch models can also be converted to TorchScript using:

torch==1.12.1 (https://pytorch.org)

torchvision==0.13.1

TorchScript models created using PyTorch 1.12 must be converted to .cat files using Nuke 14. Note that .cat files created with Nuke 14 will not run in Nuke 13.x.

For more details on TorchScript, please see the TorchScript documentation available on the PyTorch website. It may also be useful to consider the documentation specific to your PyTorch version, for example, the 1.6 documentation.

A Simple Example
In this section, we’ll create a simple object detection model called Cifar10ResNet in PyTorch. This model will detect whether or not certain objects (cat, dog, car, etc) are present in the input image. If they are, the model will return a single channel white image, otherwise it will return a single channel black image. We’ll show how this model can be converted to TorchScript, and then we’ll convert this TorchScript model to a cat file using the CatFileCreator node in Nuke. Finally, we’ll apply this model to an input image using the Inference node in Nuke.

Torchscript
Let’s start by creating our own custom Cifar10ResNet model class:

import torch
import torch.nn as nn
import torch.nn.functional as F
from resnetSmall import ResNet, ResidualBlock

class Cifar10ResNet(torch.nn.Module):
    """
    This class is a wrapper for our custom ResNet model.
    """
    def __init__(self):

        super(Cifar10ResNet, self).__init__()
        self.model = ResNet(pretrained=True)
        self.model.eval()
Here, we are defining the Cifar10ResNet class which wraps around our own pretrained ResNet model. We define Cifar10ResNet as a torch.nn.Module. Note that we are calling the ResNet model with the pretrained flag set to true so that it loads our trained network weights. We are also setting the model to evaluation mode with eval() since we are not training the model. Next, we define the model forward function:

def forward(self, input):
    """
    The forward function for this nn.Module will pass the input tensor to the ResNet
    forward function. An integer from 0 - 9 will be returned indicating which object has
    been detected. If a plane is detected, the forward function returns a tensor of ones.
    Otherwise, it returns a tensor of zeros.

    :param input: A torch.Tensor of size 1 x 3 x H x W representing the input image
    :return: A torch.Tensor of size 1 x 1 x H x W of zeros or ones
    """
    modelOutput = self.model.forward(input)
    modelLabel = int(torch.argmax(modelOutput[0]))

    plane = 0
    if modelLabel == plane:
      output = torch.ones(1, 1, input.shape[2], input.shape[3])
    else:
      output = torch.zeros(1, 1, input.shape[2], input.shape[3])
    return output
The Cifar10ResNet forward() function is a simple one that passes the input to the ResNet forward(). If the returned label is 0, a plane has been detected and the forward function returns a tensor of ones of size 1 x 1 x H x W. Otherwise, it returns a tensor of zeros of size 1 x 1 x H x W.

To convert this model to TorchScript and save it as a .pt file, simply run the following code in PyTorch:

resnet = Cifar10ResNet()
module = torch.jit.script(resnet)
module.save('cifar10_detector.pt')
Important

To create a .cat file that can be run in all versions of Nuke from Nuke 13.x onwards, the following libraries must be used when converting a PyTorch model to TorchScript:

torch==1.6 (https://pytorch.org)

torchvision==0.7

TorchScript models created using PyTorch 1.6 can be converted to .cat files using Nuke 13.x or Nuke 14. Note that TorchScript models intended to run in Nuke 13.x should be converted to .cat file using Nuke 13.x as those converted with Nuke 14 will not run in Nuke 13.x.

Nuke 14 now uses PyTorch 1.12 which means that PyTorch models can also be converted to TorchScript using:

torch==1.12.1 (https://pytorch.org)

torchvision==0.13.1

TorchScript models created using PyTorch 1.12 must be converted to .cat files using Nuke 14. Note that .cat files created with Nuke 14 will not run in Nuke 13.x.

CatFileCreator
Now that we have created the .pt file, it’s time to convert it into a .cat file that we can use in Nuke. In order to do that, launch NukeX and set up a CatFileCreator node as follows.

_images/simple-example-01.png
The Torchscript File knob points to our .pt file

The Cat File knob tells Nuke where to save our newly created .cat file.

The Channels In knob value tells Nuke that we want to process the red, green and blue channels of the image passed to the Inference node.

The Channels Out knob value ensures that our single channel output will appear in the alpha channel of the Inference node’s output image.

The Model Id allows us to encode the name of the model into the .cat file.

The scale parameter of 1 confirms that the width and height of our input image is the same as that of our output image.

Clicking the Create .cat file and Inference knob creates the .cat file and an Inference node with all of the details of the newly created .cat file.

Inference
Connect the Inference node to an image that has red, green and blue channels as follows:

_images/simple-example-02.png
The output of the model appears in the alpha channel of the Inference node’s output image. It will be all ones if a plane is detected in the input image and otherwise, it will be all zeros.

Note

This model was trained on images in sRGB colour space, so pixel values must be in sRGB before being passed to the Inference node. By default, Nuke converts all images to Nuke’s linear colour space when processing. To ensure this model works correctly, if your image was written in sRGB space, check the Raw Data knob in your Read node to ensure your pixel data doesn’t get converted to Nuke linear. Otherwise add a colourSpace node before the Inference node to convert from linear to sRGB space, and add another colour space node after the Inference node to convert pixels back to Nuke’s linear colour space.


Tensor Sizes
When writing your model in PyTorch, be aware that when this model is run inside Nuke, the input tensor to your model forward function will be of size 1 x inChan x inH x inW, where inH and inW are the height and width of the image passed to the Inference node.

Similarly, the output tensor that is returned from your model forward function must be of size 1 x outChan x outH x outW, where outH and outW are the height and width of the image output from the Inference node. inChan and outChan are the number of channels you have defined in the Channels In and Channels Out knobs in the CatFileCreator respectively.

For example, in our simple Cifar10ResNet example, the Cifar10ResNet model processed an RGB image and returned a single channel output image of ones/zeros. We set up our CatFileCreator node as follows:

_images/simple-example-01.png
When this model is run inside Nuke, the shape of the input tensor passed to the forward function:

def forward(self, input):
   """
   :param input: A torch.Tensor of size 1 x 3 x H x W representing the input image
   :return: A torch.Tensor of size 1 x 1 x H x W of zeros or ones
   """

   modelOutput = self.model.forward(input)
   modelLabel = int(torch.argmax(modelOutput[0]))

   plane = 0
   if modelLabel == plane:
     output = torch.ones(1, 1, input.shape[2], input.shape[3])
   else:
     output = torch.zeros(1, 1, input.shape[2], input.shape[3])
   return output
will be of size 1 x 3 x H x W and the output tensor returned from the forward function must be of size 1 x 1 x H x W.

If we’re using a model that processes 5 channels and outputs 3 channels, and whose Output Scale value is 2:

_images/tensor-sizes-01.png
When this model is run inside Nuke, the input tensor to the model forward function will be of size 1 x 5 x inH x inW and its output tensor must be of size 1 x 3 x 2*inH x 2*inW.


Pixel Ranges
When writing your model in PyTorch, be aware that when this model is run inside Nuke, the values in the tensor passed to your model forward function will correspond to the pixel values of the image being passed to the Inference node. This means that most values will lie in the range [0, 1], but if your input image to Inference contains superblack/superwhite pixels, your input tensor will contain values outside of the [0, 1] range.

For this reason, make sure that any preprocessing that must be performed on pixels to ensure they are in the correct range for your model is carried out in either the model forward function or externally in your Nuke script.

For example, consider the case in which a model expects all values in the input tensor to be in the range [0, 1]. Then a preprocessing step would need to be applied in the model’s forward() function to ensure that all tensor values are in the correct range:

def forward(self, input):
    """
    :param input: A torch.Tensor of size 1 x 3 x H x W representing the input image
    :return: A torch.Tensor of size 1 x 1 x H x W of zeros or ones
    """

    # This model requires tensor values to be in the
    # range [0, 1], so clamp values to this range
    input = torch.clamp(input, min = 0.0, max = 1.0)

    modelOutput = self.model.forward(input)
    modelLabel = int(torch.argmax(modelOutput[0]))

    plane = 0
    if modelLabel == plane:
      output = torch.ones(1, 1, input.shape[2], input.shape[3])
    else:
      output = torch.zeros(1, 1, input.shape[2], input.shape[3])
    return output
Alternatively, this processing could be done in the Nuke script before the image is passed to the Inference node. In this case, a Clamp node could be added before the Inference node to ensure pixels are in the correct range.

Device Assignment
In the Inference node’s properties panel you may have noticed that there is a Use GPU if available checkbox. When this box is ticked, the model and the input tensor to the forward() function will be transferred to the GPU for processing. If unticked, they will be transferred to the CPU for processing.

_images/device-assignment-01.png
In order for our .cat file to function correctly with this knob, we need to make sure that any other tensors defined in our model’s forward() function are also on the correct device.

TorchScript
Using our Cifar10ResNet example, we can edit the following lines of the forward() function to ensure that the output tensors will be created on the same device as the input tensor:

 def forward(self, input):
    """
    :param input: A torch.Tensor of size 1 x 3 x H x W representing the input image
    :return: A torch.Tensor of size 1 x 1 x H x W of zeros or ones
    """
    # Here we find out which device the input tensor is on, and assign our device accordingly
    if(input.is_cuda):
       device = torch.device('cuda')
    else:
       device = torch.device('cpu')

    modelOutput = self.model.forward(input)
    modelLabel = int(torch.argmax(modelOutput[0]))

    plane = 0
    if modelLabel == plane:
       output = torch.ones(1, 1, input.shape[2], input.shape[3], device = device)
    else:
       output = torch.zeros(1, 1, input.shape[2], input.shape[3], device = device)
    return output
This device variable should be used for all tensors created in the forward function to ensure they are created on the correct device. Having added these changes, we can create the .pt and .cat files as before.

Inference
Try toggling on and off the Use GPU if available checkbox in the Inference node to verify that the model and tensors are being transferred correctly to the given device. You should notice a large difference in computational time between the CPU and GPU.

Note
Inference currently only supports single GPU usage and not multi-GPU.


Full And Half Precision
When creating tensors in our model, we will need to make sure that they work correctly with the Optimize for Speed and Memory checkbox. When this box is ticked, the model and input tensor to the model are converted to half precision. Therefore, any other tensors defined in the model forward function will also need to be half precision. This can be done simply by assigning any tensors defined in the forward function the same dtype as the input tensor.

_images/device-assignment-01.png
TorchScript
Using our Cifar10ResNet example, we can edit the following lines to the forward function to ensure that the output tensor has the same dtype as the input tensor:

 def forward(self, input):
    """
    :param input: A torch.Tensor of size 1 x 3 x H x W representing the input image
    :return: A torch.Tensor of size 1 x 1 x H x W of zeros or ones
    """

    modelOutput = self.model.forward(input)
    modelLabel = int(torch.argmax(modelOutput[0]))

    plane = 0
    if modelLabel == plane:
       output = torch.ones(1, 1, input.shape[2], input.shape[3], dtype=input.dtype)
    else:
       output = torch.zeros(1, 1, input.shape[2], input.shape[3], dtype=input.dtype)
    return output
All tensors created in the model should be set up in this way to ensure they have the correct dtype. Having added these changes to the forward function, we can create the .pt and .cat files as before.

Inference
Try toggling on and off the Optimize for Speed and Memory checkbox in the Inference node to verify that the model and tensors are all being converted to the correct dtype.


TorchScript Type Annotation
TorchScript is statically typed, which means that variable types must be explicitly defined at compile time. For this reason, it may be necessary to annotate variable types in your model so that every local variable has a static type and every function has a statically typed signature. For more information on annotations in TorchScript, see this tutorial.

TorchScript
If we consider our simple Cifar10ResNet example, let’s add a simple normalisation function to our forward function:

def normalize(self, input, mean, std):
   """
   This method normalizes the values in input based on mean and std.
   :param input: a torch.Tensor of the size [batch x 3 x H x W]
   :param mean: A tuple of float values that represent the mean of
                the r,g,b chans e.g. (0.5, 0.5, 0.5)
   :param std: A tuple of float values that represent the std of the
               r,g,b chans e.g. (0.5, 0.5, 0.5)
   :return: a torch.Tensor that has been normalised
   """
   input[:, 0, :, :] = (input[:, 0, :, :] - mean[0]) / std[0]
   input[:, 1, :, :] = (input[:, 1, :, :] - mean[1]) / std[1]
   input[:, 2, :, :] = (input[:, 2, :, :] - mean[2]) / std[2]
   return input

def forward(self, input):
   """
   :param input: A torch.Tensor of size 1 x 3 x H x W representing the input image
   :return: A torch.Tensor of size 1 x 1 x H x W of zeros or ones
   """

   # Normalise the input tensor
   mean = (0.5, 0.5, 0.5)
   std = (0.5, 0.5, 0.5)
   input = self.normalize(input, mean, std)

   modelOutput = self.model.forward(input)
   modelLabel = int(torch.argmax(modelOutput[0]))

   plane = 0
   if modelLabel == plane:
      output = torch.ones(1, 1, input.shape[2], input.shape[3])
   else:
      output = torch.zeros(1, 1, input.shape[2], input.shape[3])
   return output
If you try to convert this model to TorchScript, you will get the following TorchScript error:

_images/type-annotation-01.png
This error indicates that when converting the model to TorchScript, the compiler assumed that the parameter mean was of type tensor, but this assumption was incorrect and caused an error. To overcome this, we need to include some annotations in the normalize() function as follows:

 def normalize(self, input, mean, std):
    """
    This method normalizes the values in input based on mean and std.
    :param input: a torch.Tensor of the size [batch x 3 x W x H]
    :param mean: A tuple of float values that represent the mean of the
                   r,g,b chans e.g. (0.5, 0.5, 0.5)
    :param std: A tuple of float values that represent the std of the
                r,g,b chans e.g. (0.5, 0.5, 0.5)
    :return: a torch.Tensor that has been normalised
    """
    # type: (Tensor, Tuple[float, float, float], Tuple[float, float, float]) -> Tensor
    input[:, 0, :, :] = (input[:, 0, :, :] - mean[0]) / std[0]
    input[:, 1, :, :] = (input[:, 1, :, :] - mean[1]) / std[1]
    input[:, 2, :, :] = (input[:, 2, :, :] - mean[2]) / std[2]
    return input
Now that these annotations have been added, this model can be successfully converted to the TorchScript format without error.

TorchScript Variables and Functions
There are several things that should be noted in relation to variables and functions when writing a model that will be converted to TorchScript. Below, we have listed some of those that arise most frequently. For more details on each of these points and others that may arise when converting your own model to TorchScript, please see the most recent TorchScript documentation as well as the TorchScript documentation specific to your PyTorch version.

External Libraries
Many of Python’s built-in functions are supported in TorchScript, along with Python’s ‘math’ module. However, no other Python modules are supported. This means that any part of your model that uses other Python modules (eg. NumPy or OpenCV ) will need to be rewritten using only functions that TorchScript supports.

Class Attribute Initialisation
Class attributes must be declared in the init function. For example, this attribute declared outside of the init function:

 class Cifar10ResNet(torch.nn.Module):
    """
    This class is a wrapper for our custom ResNet model.
    """
    def __init__(self):

       super(Cifar10ResNet, self).__init__()
       self.model = ResNet(pretrained=True)
       self.model.eval()

    def forward(self, input):
       self.label = 1
       return torch.ones(1, 1, input.shape[2], input.shape[3])
will cause an error when trying to convert this model to TorchScript. This is the correct definition process:

 class Cifar10ResNet(torch.nn.Module):
    """
    This class is a wrapper for our custom ResNet model.
    """
    def __init__(self):

       super(Cifar10ResNet, self).__init__()
       self.model = ResNet(pretrained=True)
       self.model.eval()
       self.label = 1

    def forward(self, input):
       return torch.ones(1, 1, input.shape[2], input.shape[3])
Static Variables
Unlike Python, all variables must have a single static type in TorchScript. For example, this forward function will error when converted to TorchScript since the type of variable r changes depending on the if statement:

def forward(self, input):
   if self.label == 1:
      r = 1
   else:
      r =  torch.zeros(1, 1, input.shape[2], input.shape[3])
   return input
This function should be rewritten as:

def forward(self, input):
   if self.label == 1:
      r = torch.ones(1, 1, input.shape[2], input.shape[3])
   else:
      r =  torch.zeros(1, 1, input.shape[2], input.shape[3])
   return input
Similarly, all functions must be defined so that their return variable type is clear and does not change. For example, consider a forward function defined as:

def forward(self, input):
   if self.label == 1:
      return 1
   else:
      return torch.zeros(1, 1, input.shape[2], input.shape[3])
The return type of this function can be either an integer or a tensor, depending on the if statement, and will error when converted to TorchScript. This function should be redefined, for example as:

def forward(self, input):
   if self.label == 1:
      return torch.ones(1, 1, input.shape[2], input.shape[3])
   else:
      return torch.zeros(1, 1, input.shape[2], input.shape[3])
None and Optional Types
Since TorchScript is statically typed, type annotation may be needed to ensure that variable types are correctly inferred when using None. For example, having an assignment x = None in your model will cause x to be inferred as NoneType, when it might actually be an Optional type. In this case, type annotation with x: Optional[int] = None can be used to clarify that x is indeed Optional, and can have either type integer or None.

The following code snippet indicates how to annotate an Optional class attribute in the init() function. It also demonstrates that in order to refine class attributes with Optional type outside of the init(), they must be assigned to a local variable to be refined.

 import torch
 import torch.nn as nn
 from resnet import ResNet
 from typing import Optional

 class ObjectDetection(torch.nn.Module):
    label: Optional[int]
    """
    This model detects objects in an image. If an object is detected, an image
    filled with the object label is returned. If no object is detected, an image
    of zeros is returned.
    """
    def __init__(self):
       super(ObjectDetection, self).__init__()
       self.model = ResNet(pretrained=True)
       self.model.eval()
       label = None
       self.label = label

    def forward(self, input):
       """
       This forward function updates self.label using the object label returned by the model.
       It also returns an image filled with the detected object label. This is an image
       of zeros if no object has been detected.
       """

       objectLabel = 0 # objectLabel is an integer
       output = objectLabel*torch.ones((1, 1, input.shape[2], input.shape[3]))

       # To refine self.label, its value must first be assigned to an Optional[int] variable
       # which can then be updated and reassigned to self.label
       label = self.label
       if objectLabel == 0:
          label = None
       else:
          label = objectLabel
       self.label = label

       return output
For more details on Optional type annotation, see the official TorchScript Language Reference pages version 1 and version 2.


TorchScript and Inheritance
TorchScript does not support inheritance, which means that you will have to ensure that your model classes do not have any subclass definitions. Because one model cannot be defined as a subclass of another, you can rewrite your models so that they are independent of each other. Alternatively, if you need to call one model from the forward function of another, you can define one of your models as an attribute of another, FirstModel.SecondModel, for instance.

TorchScript
Consider an object detection model that is defined with a Cifar10ResNet subclass as follows in PyTorch:

import torch
import torch.nn as nn
from resnetSmall import ResNet

class ObjectDetection(torch.nn.Module):
   def __init__(self):
      """
      This class uses a ResNet to detect 100 objects in an image. The forward function returns
      ones if an airplane is detected, and zeros otherwise.
      """
      super(ObjectDetection, self).__init__()
      self.model = ResNet()
      self.model.load_state_dict(torch.load('resnet_100.chkpt'))
      self.model.eval()

   def forward(self, input):
      """
      This forward function returns an image of ones if an airplane is detected in the image,
      and an image of zeros otherwise.
      """
      modelOutput = self.model.forward(input)
      modelLabel = int(torch.argmax(modelOutput[0]))

      if modelLabel == 1:
         output = torch.ones(1, 1, input.shape[2], input.shape[3])
      else:
         output = torch.zeros(1, 1, input.shape[2], input.shape[3])
      return output

class Cifar10ResNet(ObjectDetection):
   """
   This class detects 10 cifar objects in an image. It uses the forward function
   from the ObjectDetection class and returns an image of ones if an airplane is
   detected, and an image of zeros otherwise.
   """
   def __init__(self):
      super(Cifar10ResNet, self).__init__()
      self.model = ResNet()
      self.model.load_state_dict(torch.load('resnet_cifar10.chkpt'))
      self.model.eval()

   def forward(self, input):
      return super().forward(input)
The Cifar10ResNet model can be used for inference with the following code in PyTorch:

my_model = Cifar10ResNet()
output_image = my_model.forward(input_image)
However, this model is not convertible to TorchScript because of the class inheritance. To convert to TorchScript, the Cifar10ResNet model can be defined independently of ObjectDetection, inheriting from torch.nn.Module with its own forward function as follows:

 class Cifar10ResNet(torch.nn.Module):
    """
    This class detects 10 cifar objects in an image. Its forward function
    returns an image of ones if an airplane is detected, and an image of
    zeros otherwise.
    """
    def __init__(self):
       super(Cifar10ResNet, self).__init__()
       self.model = ResNet()
       self.model.load_state_dict(torch.load('resnet_cifar10.chkpt'))
       self.model.eval()

    def forward(self, input):

       modelOutput = self.model.forward(input)
       modelLabel = int(torch.argmax(modelOutput[0]))

       if modelLabel == 1:
          output = torch.ones(1, 1, input.shape[2], input.shape[3])
       else:
          output = torch.zeros(1, 1, input.shape[2], input.shape[3])
       return output
This model can now be converted to TorchScript as follows:

my_model = Cifar10ResNet()
module = torch.jit.script(my_model)
module.save('cifar10_resnet.pt')
On the other hand, in order to use one model in the same forward function as another, one model can be defined as an attribute of another. For example, the ObjectDetection class can instead be defined as:

 class ObjectDetection(torch.nn.Module):
    def __init__(self):
       """
       This class is a wrapper around the Cifar10ResNet  model.
       """
       super(ObjectDetection, self).__init__()
       self.Cifar10ResNet = Cifar10ResNet()

    def forward(self, input):
       output = self.Cifar10ResNet.forward(input)
       return output


 class Cifar10ResNet(torch.nn.Module):
    """
    This class is a wrapper for our custom ResNet model.
    """
    def __init__(self):
       super(Cifar10ResNet, self).__init__()
       self.model = ResNet(pretrained=True)
       self.model.eval()

    def forward(self, input):
       """
       :param input: A torch.Tensor of size 1 x 3 x H x W representing the input image
       :return: A torch.Tensor of size 1 x 1 x H x W of zeros or ones
       """
       modelOutput = self.model.forward(input)
       modelLabel = int(torch.argmax(modelOutput[0]))

       if modelLabel == 1:
          output = torch.ones(1, 1, input.shape[2], input.shape[3])
       else:
          output = torch.zeros(1, 1, input.shape[2], input.shape[3])
       return output
In this case, Cifar10ResNet is defined as an attribute of the ObjectDetection class, and its forward function can be called within the ObjectDetection forward function. The ObjectDetection model can be converted to TorchScript using the following code:

my_model = ObjectDetection()
module = torch.jit.script(my_model)
module.save('object_detection.pt')
In the section Accessing Attributes of Attributes, we will discuss how to use a custom knob in Nuke to access an attribute of a model defined as an attribute of another model.

Custom Knobs
For some models, we will want to change some of the model parameters from Nuke in order to control the output image generated by the model. For example, if we are using an object detection model, we might like to add an enumeration knob in Nuke to control which object the model is detecting in the input image. Similarly, if our model changes the colour of a person’s hair, we might like to add an enumeration knob in Nuke to control which hair colour is transferred to the person in the image.

These custom knobs can be defined in the CatFileCreator node when creating the .cat file, and can be used to control parameters defined in the model’s init() function. When this .cat file is loaded in the Inference node, all of the custom knobs will be visible and can be used to control the model output from Nuke.

In the following sections, we’ll detail how to define your model with parameters that can be controlled from Nuke, how to create custom knobs using the CatFileCreator, and how these knobs can be used in the Inference node.

TorchScript
If we consider our simple Cifar10ResNet example, let’s change the Cifar10ResNet class definition slightly so that it has an attribute called userLabel:

class Cifar10ResNet(torch.nn.Module):
    """
    This class is a wrapper for our custom ResNet model.
    """
    def __init__(self, userLabel = 0):

        super(Cifar10ResNet, self).__init__()
        self.model = ResNet(pretrained=True)
        self.model.eval()
        self.userLabel = userLabel
This is the parameter that we are going to control using a knob in Nuke. Any variables that you want to control using a knob must be declared like this, as an attribute of your model class.

Next we’ll define the model forward function as follows

def forward(self, input):
      """
      :param input: A torch.Tensor of size 1 x 3 x H x W representing the input image
      :return: A torch.Tensor of size 1 x 1 x H x W of zeros or ones
      """

      modelOutput = self.model.forward(input)
      modelLabel = int(torch.argmax(modelOutput[0]))

      if modelLabel == self.userLabel:
         output = torch.ones(1, 1, input.shape[2], input.shape[3])
      else:
         output = torch.zeros(1, 1, input.shape[2], input.shape[3])
      return output
With this new forward() function, our model will return a tensor of ones if the object defined by self.userLabel is detected, and will return a tensor of zeros otherwise. So the value stored in userLabel will define which object is detected by the model. We can use the usual steps to convert this model from Pytorch to a TorchScript .pt file.

CatFileCreator
Now that we have created a model with an attribute, we can use the CatFileCreator to define a knob to control this attribute. First, add the main CatFileCreator knob values as before:

_images/simple-example-01.png
Next, click and drag an enumeration knob to the top of the CatFileCreator node, and move to the ‘User’ tab to fill in the enumeration values as follows:

_images/custom-knobs-01.png
The most important knob here is the Name knob. The knob value you enter here MUST correspond to the name of the attribute that you want to control, as defined in your model’s init() function. In our case, this attribute is called userLabel. For our Cifar10ResNet model, label values of 0, 1, 2, 3, .., 9 correspond to objects plane, car, bird, cat, deer, etc, so these are the values we add to the Menu Items knob.

Once our enumeration knob values have been added, we’ll see the knob appear in the main CatFileCreator tab. Clicking the Create .cat file and Inference knob will create the .cat file and Inference node.

Note

Class attributes with the Optional[ ] type cannot be controlled by a custom knob in Nuke.

Only certain Nuke knobs can be added as custom knobs to the Inference node. Please see the Appendix for the list of these knobs. More details on adding knobs using the CatFileCreator can be found in the Nuke documentation.

Inference
If we open the Inference node in the properties panel, we can see that our custom Detect enumeration knob has been added. Now, if we connect the Inference node to an image of a dog, and select ‘dog’ in the Detect enumeration knob, the alpha channel returned from the Inference node will be all ones, indicating that there is a dog in the image. Changing the selected option in the enumeration knob to ‘cat’ will return an alpha channel of zeros, indicating that there is no cat in the image.

_images/custom-knobs-02.png
Note

When connecting a model attribute to a check box knob in Nuke, make sure to define the boolean attribute in your model __init__ using integer values of 0 or 1, rather than True or False:

class Cifar10ResNet(torch.nn.Module):
   def __init__(self, bool = 0):

Accessing Attributes of Attributes
TorchScript does not support inheritance, which means that you will have to ensure that your model classes do not have any subclass definitions. (See here for more details.)

Because one model cannot be defined as a subclass of another, you may find that you need to define one of your models as an attribute of another, FirstModel.SecondModel, for instance. Since Nuke does not allow knobs to have names with ‘.’ characters in them, we cannot directly connect the attributes of SecondModel to a knob in Nuke. In this case, we need to alter FirstModel to accept an argument controlling SecondModel.attribute. This method can also be used if the attribute you would like to control is nested within a torch.nn.Sequential object.

TorchScript
Consider the case in which our Cifar10ResNet class is an attribute of another class called ObjectDetection:

class ObjectDetection(torch.nn.Module):
   def __init__(self):
      """
      This class is a wrapper around the Cifar10ResNet  model.
      """
      super(ObjectDetection, self).__init__()
      self.Cifar10ResNet = Cifar10ResNet()

   def forward(self, input):
      output = self.Cifar10ResNet.forward(input)
      return output


class Cifar10ResNet(torch.nn.Module):
   """
   This class is a wrapper for our custom ResNet model.
   """
   def __init__(self):
      super(Cifar10ResNet, self).__init__()
      self.label = 1
      self.model = ResNet(pretrained=True)
      self.model.eval()

   def forward(self, input):
      """
      :param input: A torch.Tensor of size 1 x 3 x H x W representing the input image
      :return: A torch.Tensor of size 1 x 1 x H x W of zeros or ones
      """
      modelOutput = self.model.forward(input)
      modelLabel = int(torch.argmax(modelOutput[0]))

      if modelLabel == self.label:
         output = torch.ones(1, 1, input.shape[2], input.shape[3])
      else:
         output = torch.zeros(1, 1, input.shape[2], input.shape[3])
      return output
In order to use a knob to control the parameter label defined in Cifar10ResNet, we need to make the following changes to the ObjectDetection class:

 class ObjectDetection(torch.nn.Module):
    """
    This class is a wrapper around the Cifar10ResNet  model.
    """
    def __init__(self, userLabel = 0):
         super(ObjectDetection, self).__init__()
         self.Cifar10ResNet = Cifar10ResNet()
         self.userLabel = userLabel

    def forward(self, input):
         self.Cifar10ResNet.label = self.userLabel
         output = self.Cifar10ResNet.forward(input)
         return output
CatFileCreator
In CatFileCreator you can now create a custom knob and define the knob name as “userLabel”. This will allow the value of the userLabel knob to get assigned to ObjectDetection.Cifar10ResNet.label without having to use ‘.’ characters within the knob name.

_images/custom-knobs-01.png

Models with Multiple Inputs
What if your model expects two or more input images? In Nuke, only one input image can be passed to the Inference node. Therefore, only one tensor will be passed to your model forward function. If your model requires more than one input image, you will need to join your input images together to create a single image for processing.

Torchscript
Since Inference only accepts one input image, the model forward function will also only have one input tensor. Consider an example model who’s forward() function initially required two tensors, img1 and img2, as input, with each tensor representing an RGB image:

def forward(self, img1, img2):
   """
   :param img1: A torch.Tensor of size 1 x 3 x H x W representing image 1
   :param img2: A torch.Tensor of size 1 x 3 x H x W representing image 2
   :return: A torch.Tensor of size 1 x 1 x H x W of zeros or ones
   """
   output = self.model.forward(img1, img2)
   return output
In order to reformat this so that the forward() function accepts one input tensor only, we can redefine our forward function as follows:

def forward(self, input):
   """
   :param input: A torch.Tensor of size 1 x 6 x H x W representing two RGB images
   :return: A torch.Tensor of size 1 x 1 x H x W of zeros or ones
   """

   # Split the input tensor into two tensors,
   # one representing each image
   input_split = torch.split(input, 3, 1)
   img1 = input_split[0]
   img2 = input_split[1]

   output = self.model.forward(img1, img2)
   return output
where input now has 6 channels and can be split into two tensors, img1 and img2, before passing them to model.forward().

Channel Shuffling
Since the Inference node only accepts one input image, channels from images that you would like processed by the model will have to be shuffled into a single image in Nuke. Since our model forward function now accepts a 6 channel tensor, we can use a shuffle node in Nuke to shuffle two RGB images into a single image with 6 channels - rgba.red, rgba.green, rgba.blue, forward.u, forward.v, and backward.u, for example - and pass this image as input to the Inference node.

_images/multiple-inputs-01.png _images/multiple-inputs-02.png
CatFileCreator
When converting this model from TorchScript to a .cat file, the Channels In knob should define the 6 channels from the input image to Inference that you would like to process by the model. In our case, the 6 channels that we would like to process are the rgba.red, rgba.green, rgba.blue, forward.u, forward.v, and backward.u channels of the input image, so we define the CatFileCreator knobs as follows:

_images/multiple-inputs-03.png
Note

Any of the default Nuke channels or any channels that are already defined in your script can be used in the Channels In and Channel Outs knobs of the CatFileCreator. See the appendix for the full list of Nuke channel names that can be used with these knobs.

Inference
When we have created the Inference node, we can see that the 6 channels from the input image that will be processed by the model are listed in Channels In:

_images/multiple-inputs-04.png
This indicates that we need to pass an image to this Inference node that has all 6 of these channels. We can connect this Inference node to the 6 channel image we created earlier to get the final output.

Models with Multiple Outputs
What if your model returns two or more output images? In Nuke, only one output image will be returned from the Inference node. Therefore, only one tensor can be returned from your model forward function. If your model returns more than one output image, you will need to join your output images together to create a single image to return.

Torchscript
Since Inference only returns one output image, the model forward() function will also only return one output tensor. Consider an example model who’s forward function returns two RGB tensors, img1 and img2, as output:

def forward(self, input):
   """
   :param input: A torch.Tensor of size 1 x 3 x H x W
   :return img1: A torch.Tensor of size 1 x 3 x H x W
   :return img2: A torch.Tensor of size 1 x 3 x H x W
   """
   [img1, img2] = self.model.forward(input)
   return img1, img2
This forward() function can be redefined as follows to ensure that it returns one tensor only:

def forward(self, input):
   """
   :param input: A torch.Tensor of size 1 x 3 x H x W
   :return: A torch.Tensor of size 1 x 6 x H x W
   """
   [img1, img2] = self.model.forward(input)
   output = torch.cat((img1, img2), 1)
   return output
where output now has 6 channels.

CatFileCreator
Now that our model is defined so that the forward function returns 6 channels, we use the Channels Out knob in the CatFileCreator node to define where we want these 6 channels to appear in the output image. For example, setting Channels Out as:

_images/multiple-outputs-01.png
will ensure that img1 will appear in the rgba.red, rgba.green and rgba.blue channels while img2 will appear in the forward.u, forward.v and backward.u channels.

Note

Any of the default Nuke channels or any channels that are already defined in your script can be used in the Channels In and Channel Outs knobs of the CatFileCreator.

Inference
Having created the .cat file, we can connect it to the input image and see that the output contains the rgba.red, rgba.green, rgba.blue, forward.u, forward.v and backward.u channels. We can use two shuffle nodes after the Inference node to split this output image back up into two RGB images:

_images/multiple-outputs-02.png

Putting it all Together: Example 1
Now that we have discussed the topics that you should bear in mind when writing a model and converting it from PyTorch to a .cat file, let’s update our simple Cifar10ResNet model to a more advanced version.

TorchScript
Let’s define our Cifar10ResNet model as follows:

import torch
import torch.nn as nn
import torch.nn.functional as F
from resnetSmall import ResNet, ResidualBlock

class Cifar10ResNet(torch.nn.Module):
    """
    This is a wrapper class for our custom ResNet model. Its purpose is to
    preprocess the input tensors and ensure that all tensors are created on
    the correct device with the correct dtype. It also defines the variable
    userLabel as an attribute that can be controlled by a custom knob in Nuke.
    """
    def __init__(self, userLabel = 0):
        """
        :param userLabel: This int controls which object is detected by the
                          Cifar10ResNet model and can be linked to an enumeration
                          knob in Nuke.
        """
        super(Cifar10ResNet, self).__init__()
        self.model = ResNet(pretrained=True)
        self.model.eval()
        self.userLabel = userLabel
Note that this class defines the attribute userLabel which can be controlled by a custom knob in Nuke.

Let’s define our model’s normalize() and forward() functions as follows:

def normalize(self, input, mean, std):
      """
      This method normalizes the values in input based on mean and std.
      :param input: a torch.Tensor of the size [batch x 3 x H x W]
      :param mean: A tuple of float values that represent the mean of the
                           r,g,b chans e.g. [0.5, 0.5, 0.5]
      :param std: A tuple of float values that represent the std of the
                        r,g,b chans e.g. [0.5, 0.5, 0.5]
      :return: a torch.Tensor that has been normalised
      """
      # type: (Tensor, Tuple[float, float, float], Tuple[float, float, float]) -> Tensor
      input[:, 0, :, :] = (input[:, 0, :, :] - mean[0]) / std[0]
      input[:, 1, :, :] = (input[:, 1, :, :] - mean[1]) / std[1]
      input[:, 2, :, :] = (input[:, 2, :, :] - mean[2]) / std[2]
      return input

def forward(self, input):
      """
      The forward function for this model will normalize the input and then pass it
      to the Resnet forward function. It will compare the returned label to userLabel label and
      return either an all ones or all zeros tensor.
      :param input: A torch.Tensor of size 1 x 3 x H x W representing the input image
      :return: A torch.Tensor of size 1 x 1 x H x W of zeros or ones
      """
      # Determine which device all tensors should be created on
      if(input.is_cuda):
         device = torch.device('cuda')
      else:
         device = torch.device('cpu')

      # Normalise the input tensor
      mean = (0.5, 0.5, 0.5)
      std = (0.5, 0.5, 0.5)
      input = self.normalize(input, mean, std)

      modelOutput = self.model.forward(input)
      modelLabel = int(torch.argmax(modelOutput[0]))

      # Check if the detected object is the same as userLabel
      if modelLabel == self.userLabel:
         # Ensure output is created on the correct device with the correct dtype
         output = torch.ones((1, 1, input.shape[2], input.shape[3]), dtype = input.dtype, device=device)
      else:
         # Ensure output is created on the correct device with the correct dtype
         output = torch.zeros((1, 1, input.shape[2], input.shape[3]), dtype = input.dtype, device=device)
      return output
Note that the normalize() function is applied to the input tensor to ensure tensor values are in the range expected by the model. When defining the normalize() function, we use annotations to clarify what the input and output parameter types are. The forward function accepts an input tensor of size 1 x 3 x H x W, and outputs a tensor of size 1 x 1 x H x W. In the forward function, we also ensure that all tensors are created on the correct device, with the correct dtype.

We can convert this model to a TorchScript file using the following code

resnet = Cifar10ResNet()
module = torch.jit.script(resnet)
module.save('cifar10_resnet.pt')
CatFileCreator
In the CatFileCreator node in Nuke, set the default knobs as follows:

_images/simple-example-01.png
Next, add an enumeration knob by dragging and dropping it to the top of the CatFileCreator properties panel. Fill in the enumeration knob values as follows:

_images/custom-knobs-01.png
This will create an enumeration knob in your CatFileCreator node. Next, click the Create .cat file and Inference to create your .cat file and prepopulated Inference node.

Inference
Opening the new Inference node’s properties panel, you will see the knob values have been prepopulated and the Detect enumeration knob has been created. The Inference node can be connected to an image with red, green and blue channels to get the output result:

_images/custom-knobs-02.png
Note that the Cifar10ResNet model was trained in sRGB space. Therefore, to ensure our input image is in the correct colour space, we have checked the Raw Data knob in the Read node since this image was originally rendered in the sRGB space. Alternatively, a colour space node can be added before the Inference node to convert the image from Nuke linear to sRGB colour space.

Now that the input image is connected, change the object selected in the enumeration knob to see the effect it has on the output image. If the input image contains the selected object, the alpha channel of the output image will contain all ones, otherwise it will contain all zeros. Toggle the Use GPU if available and Optimise for speed and Memory knobs to confirm that they are also working as expected.

Putting it all Together: Example 2
Our second example is a simple colour transfer model which transfers the colour distribution of one image to another. This example will highlight:

how to pass multiple images to your model

it will show that normalisation functions can be used in your model’s forward() function to ensure pixels are in the correct range

it will show how to control the attributes of a model that is defined as an attribute of another model using a float valued knob in Nuke

The parent model defined in this example, ColourTransfer, is a wrapper around another model, LinearColourTransfer. It is the LinearColourTransfer model that contains the main functionality for our colour transfer method, which is based on a simple linear transformation.

TorchScript
Let’s start by defining our LinearColourTransfer model, which contains the main colour transfer functionality

import torch
import torch.nn as nn

class LinearColourTransfer(nn.Module):
   """This model transfers the color distribution from one image to another
      using a linear transformation. The variable 'mix' controls how much the
      colours of the first image are changed to look like the second image.
   """
   def __init__(self):
      super(LinearColourTransfer, self).__init__()
      self.linear_layer = nn.Linear(6, 3)
      self.mix = 1.0
Note that we have defined our model with an attribute mix, which will control how much the colour in the first RGB image will be changed. This is a floating point value so we define it as mix = 1.0.

Next, let’s define our model’s forward() function as:

def forward(self, input):
      """

      This forward function accepts an input tensor representing two RGB images. The first RGB
      image will be recoloured so that it matches the color distribution of the second.

      :param input: A torch.Tensor of size 1 x 6 x H x W representing two RGB images
      :return: A torch.Tensor of size 1 x 3 x H x W representing the first RGB image,
                  recoloured so that it has the same mean and std deviation as the second image
      """
      # Normalize the first image stored in the tensor
      norm_input = input.clone()
      norm_input[:, 0:3, :, :] = self.normalize(input[:, 0:3, :, :])

      # Reshape the input tensor so it has size [(H * W) x 6]
      # This is the size required by self.linear_layer
      b, c, h, w = norm_input.size()
      reshaped_input = torch.reshape(norm_input, (6, h*w))
      reshaped_input = torch.transpose(reshaped_input, 0, 1)

      # Apply the linear colour transformation to the first image using the
      # colour distribution of the second image
      transformed_input = self.linear_layer(reshaped_input)

      # Reshape the output tensor so it has size [1 x 3 x H x W]
      transformed_input = torch.transpose(transformed_input, 0, 1)
      transformed_input = torch.reshape(transformed_input, (3, h, w))
      transformed_input = torch.unsqueeze(transformed_input, dim = 0)


      # Using .transpose() may have altered the tensor in memory so that it is no longer
      # contiguous, so apply the contiguous() function to fix this
      transformed_input = transformed_input.contiguous()

      # Use the 'mix' variable to control how different the final image will be to the input
      output = (1 - self.mix)*input[:, 0:3, :, :] + (self.mix)*transformed_input

      return output
Since this model uses two RGB images, its forward function accepts an input tensor of size 1 x 6 x H x W. Since self.linear_layer requires an input of size (H*W) x 6, this input tensor is reshaped before and after self.linear_layer is applied. This also ensures that the forward function’s output tensor has the expected shape of 1 x 3 x H x W. Note that when reshaping the tensors, using transpose() can create a tensor that is not contiguous in memory, so we need to call contiguous() on the transposed tensor to combat this.

The first input image is also normalised to ensure that it has the pixel range expected by self.linear_layer. The definition of normalise() can be found in the supporting python files.

Finally, in the final line, the mix variable is used to control how much the original and recoloured image are mixed in the output. This is the variable that will be controlled later by a custom knob in Nuke.

Next, let’s define the parent model, ColourTransfer, that wraps around this LinearColourTransfer model:

class ColourTransfer(nn.Module):
   """This model is a wrapper around our LinearColourTransfer Model.
   """
   def __init__(self, mixValue = 1.0):
      super(ColourTransfer, self).__init__()
      self.mixValue = mixValue
      self.LinearColourTransfer = LinearColourTransfer()

      # Load weights for LinearColourTransfer
      checkpoint_file = 'colourtransfer_ckpt_600.pth'
      checkpoint = torch.load(checkpoint_file)
      self.LinearColourTransfer.load_state_dict(checkpoint)
      self.LinearColourTransfer.eval()

   def forward(self, input):
      """
      :param input: A torch.Tensor of size 1 x 6 x H x W representing two RGB images
      :return: A torch.Tensor of size 1 x 3 x H x W representing the recoloured image
      """
      # Control self.LinearColourTransfer.mix using mixValue
      self.LinearColourTransfer.mix = self.mixValue

      output = self.LinearColourTransfer(input)

      return output
Since this is the parent model, the attributes that are declared in this class are the ones that we can control from Nuke with custom knobs, so we define a mixValue attribute in the __init__ function. The first line of this model’s forward() function ensures that mixValue controls LinearColourTransfer.mix.

In the __init__ function, we also load the weights for LinearColourTransfer and set the model to inference mode with .eval().

Now that we have defined our main ColourTransfer model, we can run the following code to convert this model to a TorchScript file:

model = ColourTransfer()
scripted_model = torch.jit.script(model)
scripted_model.save('colour_transfer.pt')
CatFileCreator
In the CatFileCreator node in Nuke, set the default knobs as follows:

_images/example2-01.png
Since our model requires two RGB images as input and returns an RGB image as output, we’ve defined 6 input channels in the Channels In knob and three output channels in the Channels Out knob.

Next, add a float knob by dragging and dropping it to the top of the CatFileCreator properties panel and fill in the knob values as follows:

_images/example2-02.png
Note that the Name knob value is set to ‘mixValue’, the attribute defined in our parent ColourTransfer model. Click Create .cat file and Inference to create your .cat file and prepopulated Inference node.

Inference
Opening the new Inference node’s properties panel, you will see the knob values have been prepopulated and the Mix float knob has been created. From the Channels In knob we can see that this Inference node can be connected to an image containing the rgba.red, rgba.green, rgba.blue, forward.u, forward.v, backward.u channels. We create our 6 channel input image by reading in two RGB images and combining them by shuffling the RGB channels of the second image into the forward.u, forward.v and backward.u channels of the first as follows:

_images/example2-03.png
The Inference node can be attached to this image and will return the first RGB image recoloured with the colours of the second RGB image:

_images/example2-04_resize.png
Note that the LinearColourTransfer model was trained in sRGB space. Therefore, to ensure our input image is in the correct colour space, we have checked the Raw Data knob in both Read nodes since these images were originally rendered in the sRGB space. Alternatively, a colour space node can be added before the Inference node to convert the image from Nuke linear to sRGB colour space.

Now that the input image is connected, change the value of the custom Mix knob to see the effect it has on the output image. It will control how much the output image mixes between the original image and the recoloured image. Toggle the Use GPU if available and Optimise for speed and Memory knobs to confirm that they are also working as expected.

Note

If instead you wanted to store the first RGB image in the motion channels and the second RGB image in the RGB channels, you can. In the CatFileCreator, just define the Channels In knob value as ‘forward.u, forward.v, backward.u, rgba.red, rgb.green, rgba.blue’ and ensure that you shuffle the first image into the motion channels and the second into the RGB channels.

   






