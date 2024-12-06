**Modified U-Net model -- attention U-Net with residual blocks**

This folder contains overall information and codes for employing a modified U-Net model for image segmentation.

**GOAL:**

The aim is to perform binary image segmentation to accurately separate a single breast tumor from the background, i.e. rest of the tissue and body parts, in ultrasound images. 

**CODE:**

*code_U-net_model.py*
This file contains the code of a traditional U-Net model, an architecture commonly used in the field of medicine for image segmentation which has been modified by incorporating attention gates and residual blocks to improve its performance. 
This code can be run with the following line:

*Prepare_dataset.py*
This file contains a code to upload a set of images for training and testing the models. Images, which later will be pre-processed (resizing, normalization, augmentation), can be uploaded with the following line:

Note than an output directory for the segmentation of test images must be uploaded in this line:

*training&testing.py*
This file contains the code for training and testing the U-Net model, it can be run with the following line:
