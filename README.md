# **Breast Ultrasound Image Segmentation Using Deep Learning for Accurate Diagnosis of Breast Lesions**

**Supervisor(s):** Zhikai Yang  
**Flavor/Track:** Imaging / Computer Science  


## **Background**

Breast cancer remains one of the most prevalent cancers impacting women globally. Early detection and diagnosis are critical in enhancing treatment success and survival rates. Breast ultrasound imaging, as a non-invasive, cost-effective technique, is commonly used for breast lesion detection. However, manual interpretation of ultrasound images requires significant radiologist expertise, is time-consuming, and subject to errors and variability.

This project focuses on automating the segmentation of breast lesions to assist radiologists in more accurate and efficient image analysis. Despite advancements in deep learning for medical image segmentation, challenges remain due to the complex nature of breast ultrasound images, such as artifacts, low contrast, and irregular lesion shapes. This research aims to explore and develop deep learning-based methods to accurately segment breast ultrasound images.


## **Project Goals**

- Develop a robust and efficient deep learning model for the accurate segmentation of breast lesions in ultrasound images.
- Evaluate the model on publicly available breast ultrasound datasets and compare its performance against baseline models.
- *(Optional)* Explore techniques such as transfer learning, multi-scale feature extraction, and attention mechanisms to enhance segmentation accuracy.


## **Tasks and Starting Points**

- **Data Collection**: Use publicly available breast ultrasound image datasets, such as the BUSI (Breast Ultrasound Image) dataset, which includes labeled images of benign, malignant, and normal tissue.
  
- **Data Preprocessing**: Preprocess images to normalize intensity values and reduce noise. Apply data augmentation techniques (e.g., rotation, flipping, contrast adjustment) to expand the training set and increase model robustness.
  
- **Model Architecture**: Employ a modified UNet model with residual blocks and attention modules to enhance segmentation accuracy. Further exploration will include feature pyramids and multi-scale extraction to capture both fine details and global context.
  
- **Training**: Train the model using a combination of cross-entropy loss and Dice coefficient to address class imbalance. Leverage transfer learning from pre-trained models (e.g., ResNet) to improve feature extraction.
  
- **Evaluation**: Assess the model with metrics like Intersection over Union (IoU), Dice coefficient, sensitivity, and specificity. Baseline comparisons will include the classical UNet and other popular segmentation models.


## **References**

- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*.
- Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. *CVPR*.
- Litjens, G., Kooi, T., Bejnordi, B. E., et al. (2017). A Survey on Deep Learning in Medical Image Analysis. *Medical Image Analysis*.
