import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

# 输入和输出路径
input_dir = '/root/Dataset/BUS_Mix_Seg_512/labels'   # 原始图像路径
output_dir = '/root/Dataset/BUS_Mix_Seg_resize_512/labels'  # 保存路径
target_size = (512, 512)

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

def pad_to_square(img):
    """将图像填充为正方形（保持内容比例）"""
    width, height = img.size
    
    # 确定要填充的尺寸
    max_size = max(width, height)
    
    # 创建一个新的正方形背景图像（黑色背景）
    square_img = Image.new('RGB', (max_size, max_size), (0, 0, 0))
    
    # 计算粘贴位置（居中）
    paste_x = (max_size - width) // 2
    paste_y = (max_size - height) // 2
    
    # 将原图粘贴到正方形背景上
    square_img.paste(img, (paste_x, paste_y))
    
    return square_img

# 主循环
for fname in tqdm(os.listdir(input_dir)):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    in_path = os.path.join(input_dir, fname)
    out_path = os.path.join(output_dir, fname)

    try:
        img = Image.open(in_path)
        
        # 如果图像有透明通道，转换为RGB
        if img.mode == 'RGBA':
            img = img.convert('RGB')
            
        # 1. 将图像填充为正方形（保持内容比例）
        square_img = pad_to_square(img)
        
        # 2. 调整大小到目标尺寸
        resized = square_img.resize(target_size, Image.BILINEAR)
        
        # 保存结果
        resized.save(out_path)
    except Exception as e:
        print(f"[ERROR] Failed to process {fname}: {e}")
