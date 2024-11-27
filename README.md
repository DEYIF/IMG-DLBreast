
# **Project Basic Info**
Breast Ultrasound Image Segmentation Using Deep Learning for Accurate Diagnosis of Breast Lesions

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


# <h1 align="center">● Medical SAM Adapter</h1>
<p align="center">
    <a href="https://discord.gg/DN4rvk95CC">
        <img alt="Discord" src="https://img.shields.io/discord/1146610656779440188?logo=discord&style=flat&logoColor=white"/></a>
    <img src="https://img.shields.io/static/v1?label=license&message=GPL&color=white&style=flat" alt="License"/>
</p>

Medical SAM Adapter, or say MSA, is a project to fineturn [SAM](https://github.com/facebookresearch/segment-anything) using [Adaption](https://lightning.ai/pages/community/tutorial/lora-llm/) for the Medical Imaging.
This method is elaborated on the paper [Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation](https://arxiv.org/abs/2304.12620).

## Original Github Repository `https://github.com/SuperMedIntel/Medical-SAM-Adapter`

## Check this document for more instructions [Notion | SAM Code](https://www.notion.so/SAM-Doc-121a5f3a96a680598c5de66fe386518c?pvs=4)

---

## A Quick Overview 
 <img width="880" height="380" src="https://github.com/WuJunde/Medical-SAM-Adapter/blob/main/figs/medsamadpt.jpeg">


 ## Requirement

 Install the environment:

 ``conda env create -f environment.yml``

 ``conda activate sam_adapt``

 Then download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth), and put it at ./checkpoint/sam/

 You can run:

 ``wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth``

 ``mv sam_vit_b_01ec64.pth ./checkpoint/sam``
 creat the folder if it does not exist

 ## Example Cases

 ### Melanoma Segmentation from Skin Images (2D)

 1. Download ISIC dataset part 1 from https://challenge.isic-archive.com/data/. Then put the csv files in "./data/isic" under your data path. Your dataset folder under "your_data_path" should be like:
ISIC/
     ISBI2016_ISIC_Part1_Test_Data/...
     
     ISBI2016_ISIC_Part1_Training_Data/...
     
     ISBI2016_ISIC_Part1_Test_GroundTruth.csv
     
      ISBI2016_ISIC_Part1_Training_GroundTruth.csv
    
    You can fine the csv files [here](https://github.com/KidsWithTokens/MedSegDiff/tree/master/data/isic_csv)

 1. Begin Adapting! run: ``python train.py -net sam -mod sam_adpt -exp_name *msa_test_isic* -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 32 -dataset isic -data_path *../data*``
 change "data_path" and "exp_name" for your own useage. you can change "exp_name" to anything you want.

 You can descrease the ``image size`` or batch size ``b`` if out of memory.

 3. Evaluation: The code can automatically evaluate the model on the test set during traing, set "--val_freq" to control how many epoches you want to evaluate once. You can also run val.py for the independent evaluation.

 4. Result Visualization: You can set "--vis" parameter to control how many epoches you want to see the results in the training or evaluation process.

 In default, everything will be saved at `` ./logs/`` 

## Run on  your own dataset (Change different Datasets)
It is simple to run MSA on the other datasets. Just write another dataset class following which in `` ./dataset.py``. You only need to make sure you return a dict with 
     {
                 'image': A tensor saving images with size [C,H,W] for 2D image, size [C, H, W, D] for 3D data.
                 D is the depth of 3D volume, C is the channel of a scan/frame, which is commonly 1 for CT, MRI, US data. 
                 If processing, say like a colorful surgical video, D could the number of time frames, and C will be 3 for a RGB frame.
                 'label': The target masks. Same size with the images except the resolutions (H and W).
                 'p_label': The prompt label to decide positive/negative prompt. To simplify, you can always set 1 if don't need the negative prompt function.
                 'pt': The prompt. Should be the same as that in SAM, e.g., a click prompt should be [x of click, y of click], one click for each scan/frame if using 3d data.
                 'image_meta_dict': Optional. if you want save/visulize the result, you should put the name of the image in it with the key ['filename_or_obj'].
                 ...(others as you want)
     }
Welcome to open issues if you meet any problem. It would be appreciated if you could contribute your dataset extensions. Unlike natural images, medical images vary a lot depending on different tasks. Expanding the generalization of a method requires everyone's efforts.

 ## Cite
 ~~~
@misc{wu2023medical,
      title={Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation}, 
      author={Junde Wu and Wei Ji and Yuanpei Liu and Huazhu Fu and Min Xu and Yanwu Xu and Yueming Jin},
      year={2023},
      eprint={2304.12620},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
 ~~~





