## Modified U-Net model -- attention U-Net with residual blocks

This folder contains overall information and codes for employing a modified U-Net model for image segmentation.

**GOAL:**

The aim is to perform binary image segmentation to accurately separate a single breast tumor from the background, i.e. rest of the tissue and body parts, in ultrasound images. 

**CODE:**

*code_U-net_model.py*
This file contains the code of a traditional U-Net model, an architecture commonly used in the field of medicine for image segmentation which has been modified by incorporating attention gates and residual blocks to improve its performance. 
This code can be run with the following line:

*Prepare_dataset.py*
This script allows to upload customized paths for your images and labels, later pre-processed (resizing, normalization and augmentation) and used for training and testing the model. These images and labels can be uploaded following these instructions:

1. When you run the script, you will be prompted to input paths for the image and label directories. 

2. Simply copy and paste your desired paths when prompted. If you want to use the default paths, press **Enter** without typing anything.

*training&testing.py*
This file contains the code for training and testing the U-Net model, it can be run with the following line:
