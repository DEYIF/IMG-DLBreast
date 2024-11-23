import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# Feature Extraction Functions

def compute_margin_features(contour):
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    smoothness = np.std([cv2.pointPolygonTest(contour, (int(point[0][0]), int(point[0][1])), True) for point in approx])
    angles = []
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i+1) % len(approx)][0]
        p3 = approx[(i+2) % len(approx)][0]
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(angle)
    spikiness = np.sum(np.array(angles) < np.pi / 6)
    return smoothness, spikiness

def compute_shape_features(contour):
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (center, axes, orientation) = ellipse
        major_axis, minor_axis = max(axes), min(axes)
        aspect_ratio = major_axis / minor_axis
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if aspect_ratio < 1.2 and circularity > 0.75:
            shape = "Round"
        elif aspect_ratio < 1.5 and circularity > 0.6:
            shape = "Oval"
        else:
            shape = "Irregular"
        return shape, aspect_ratio, circularity
    else:
        return "Irregular", None, None

def compute_posterior_acoustic_features(contour, gray_image):
    x, y, w, h = cv2.boundingRect(contour)
    roi = gray_image[y + h:y + 2 * h, x:x + w]
    if roi.size == 0:
        return "No Posterior Acoustic Features", None, None, roi
    mean_intensity = np.mean(roi)
    std_intensity = np.std(roi)
    enhancement_threshold = 80
    shadowing_threshold = 60
    if mean_intensity > enhancement_threshold:
        posterior_feature = "Posterior Acoustic Enhancement"
    elif mean_intensity < shadowing_threshold:
        posterior_feature = "Posterior Acoustic Shadowing"
    else:
        posterior_feature = "No Posterior Acoustic Features"
    return posterior_feature, mean_intensity, std_intensity, roi

def compute_glcm_features(contour, gray_image):
    x, y, w, h = cv2.boundingRect(contour)
    roi = gray_image[y:y + h, x:x + w]
    if roi.size == 0:
        return "Complex", None, None, None, None
    glcm = graycomatrix(roi, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    if energy > 0.3 and homogeneity > 0.3:
        echo_pattern = "Anechoic"
    elif contrast < 1000 and correlation > 0.5:
        echo_pattern = "Hypoechoic"
    elif contrast < 2000 and energy > 0.1:
        echo_pattern = "Isoechoic"
    elif contrast >= 2000 or correlation < 0.5:
        echo_pattern = "Hyperechoic"
    else:
        echo_pattern = "Complex"
    return echo_pattern, contrast, correlation, energy, homogeneity

def compute_lesion_boundary_feature(contour, gray_image, k=5):
    contour_points = contour.reshape(-1, 2)
    distance_map = np.zeros_like(gray_image, dtype=np.uint8)
    for point in contour_points:
        distance_map[point[1], point[0]] = 1
    distance_map = cv2.distanceTransform(1 - distance_map, cv2.DIST_L2, 5)
    surrounding_tissue = []
    outer_mass = []
    for y in range(gray_image.shape[0]):
        for x in range(gray_image.shape[1]):
            if distance_map[y, x] < k:
                surrounding_tissue.append(gray_image[y, x])
            elif distance_map[y, x] < 2 * k:
                outer_mass.append(gray_image[y, x])
    avg_tissue_intensity = np.mean(surrounding_tissue) if surrounding_tissue else 0
    avg_mass_intensity = np.mean(outer_mass) if outer_mass else 0
    lbd = avg_tissue_intensity - avg_mass_intensity
    return lbd

# Processing Single Image Function

import cv2
import pandas as pd
import matplotlib.pyplot as plt

# 读取 图像 文件
image_folder = '/content/drive/MyDrive/BUS/train/images'
label_folder = '/content/drive/MyDrive/BUS/train/labels'
import os

# 存储结果的列表
results = []

# 遍历所有图像 read all datas
for filename in os.listdir(image_folder):
  #if filename.endswith(".png") or filename.endswith(".jpg"):
    # 构建图像和掩码的路径 Paths to build images and masks
    image_path = os.path.join(image_folder, filename)
    mask_path = os.path.join(label_folder, filename)
    
    # 读取图像和掩码 read the image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
      print(f"can't read mask: {mask_path}")
      continue
    # 转为灰度图像 Grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊，减少噪声 Gaussian blur , reduces noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # 确保掩码是二值化的 Make sure the mask is binary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(binary_mask) == 0:
      print(f"mask doesn't have valid value：{mask_path}")
      continue
    # 找到掩码中的轮廓 Find the outline in the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 假设最大轮廓是病灶 Assume that the maximum contour is the lesion
    lesion_contour = max(contours, key=cv2.contourArea)
    
    # 可视化轮廓 Visual contour
    contour_image = image.copy()
    cv2.drawContours(contour_image, [lesion_contour], -1, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.title("Lesion Contour")
    plt.show()
    #add crop
    crop_value = filename
    
    # 计算所有特征 calculate all the features
    shape, aspect_ratio, circularity = compute_shape_features(lesion_contour)
    posterior_feature, mean_intensity, std_intensity, roi = compute_posterior_acoustic_features(lesion_contour, gray_image)
    smoothness, spikiness = compute_margin_features(lesion_contour)
    echo_pattern, contrast, correlation, energy, homogeneity = compute_glcm_features(lesion_contour, gray_image)
    lbd = compute_lesion_boundary_feature(lesion_contour, gray_image, k=5)
    
    # 输出结果 print result
    result = {
        'filename': filename ,
        'crop': crop_value,
        'shape': shape ,
        'Aspect Ratio': aspect_ratio,
        'Circularity': circularity,
        'posterior_feature': posterior_feature,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'smoothness': smoothness,
        'spikiness': spikiness,
        'echo_pattern': echo_pattern,
        'Contrast': contrast,
        'Correlation': correlation,
        'Energy': energy,
        'Homogeneity': homogeneity,
        'lbd': lbd
    }
    print(result)
    results.append(result)
  #else:
    #print("invalid mask")


import os
import pandas as pd

# 将结果保存到 CSV 文件 save result in csv document
result_csv_path = '/content/drive/MyDrive/BUS/Total_train.csv'
results_df = pd.DataFrame(results)
results_df.to_csv(result_csv_path, index=False)

# 检查文件是否存在 check if the document exist
if os.path.exists(result_csv_path):
    print(f"Processing complete. Results saved to '{result_csv_path}'.")
else:
    print("fail to save")


#results_df = pd.DataFrame(results)
#results_df.to_csv('Total_train.csv', index=False)
#print("Processing complete. Results saved to 'All_Train'.")

