# eva8_codebase

### Main Code Repository

1. This repository contains the main code that could be used for EVA8
2. It contains the following modules and packages

### models
This package contains the following:
1. ```__init__.py``` : the init file that makes it easy to import the modules in this ```models``` package.
2. ```resnet.py```: this module contains the code for ```ResNet18``` and ```ResNet34``` models.


### utils
This package contains the following:
1. ```__init__.py```: the init file that makes it easy to import the modules in this ```utils``` package.
2. ```data_utils.py```: this module contains:
    1. ```Cifar10Dataset``` class to create the CIFAR-10 datasets with the given ```train``` and ```test``` transforms.
    2. ```get_train_transforms()``` and ```get_test_transforms()``` functions to get the ```train``` and ```test``` transformations from ```albumentations``` module.
    3. ```get_device()``` method to get and set the ```cuda``` or ```CPU``` device.
    4. ```get_dataloader_args()``` function to get the data loader arguments. This function can generate dataloader arguments (based on device: cpu/cuda).
    5. ```load_data()``` function takes the dataloader arguments and generates the ```train/test``` data loader with ```transformations```.

3. ```plots.py```: This module contains:
    1. ```model_summary()```: Function to give the summary of the model.
    2. ```plot_stats()```: Function to plot the ```Loss``` and ```Accuracy``` or other metrics that have been captured during ```training/testing``` phase.
    3. ```plot_imgs```: Function to plot the ```misclassified``` images by the model. Usually ```test dataloader``` will be used to test the model.

4. ```grad_cam.py```: This module contains:
    1. ```get_cam()``` : Function to generate the ```GradCAM``` object for producing the ```class activation maps```
    2. ```show_cam_on_image()```: Function to overlay the mask on the image(test image) that has been misclassified.
    3. ```plot_cam()```: Function to plot the ```misclassified image```, ```mask/class activation map``` of the test image.



### main.py
1. ```Trainer``` class that's the base class to generate the ```trainer``` that can do the following:
    1. ```train```: method to ```train``` the model with ```trainloader```.
    2. ```test```: method to ```test``` the model with ```testloader```.
    3. ```get_train_stats```: method to get the ```loss, accuracy and other metrics``` during the training.
    4. ```get_test_stats```: method to get the ```loss, accuracy and other metrics``` during the test.
    5. ```get_misclassified_images```: method to get the misclassified images from the ```testloader```
    6. ```get_criterion_for_classification```: method to get the ```Loss criterion``` for the ```classfication``` problem.
    7. ```get_sgd_optimizer``` and ```get_adam_optimizer```: method to get the required optimizer. (In future will be moved to ```optim.py``` module that contains that optimizers and schedulers.)

