import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

# 输入和输出路径
input_dir = '/root/Dataset/HMSS_US Cases NL/images'   # 原始图像路径
output_dir = '/root/Dataset/HMSS_512/test/images'  # 保存路径
target_size = (512, 512)

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

def center_top_crop_square(img):
    """对普通图像执行中间偏上的正方形裁剪"""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = int((h - side) * 0.25)
    top = max(0, min(top, h - side))
    return img.crop((left, top, left + side, top + side))

def crop_largest_square_from_alpha(img):
    if img.mode == 'RGBA':
        alpha = img.getchannel('A')
        alpha_np = np.array(alpha)
        ys, xs = np.where(alpha_np > 0)
        if len(xs) == 0 or len(ys) == 0:
            return center_top_crop_square(img.convert('RGB'))

        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        box_w, box_h = x2 - x1, y2 - y1
        side = min(box_w, box_h)

        left = x1 + (box_w - side) // 2
        top = y1 + int((box_h - side) * 0.25)
        left = max(0, left)
        top = max(0, top)
        right = left + side
        bottom = top + side

        return img.crop((left, top, right, bottom)).convert('RGB')
    else:
        return center_top_crop_square(img)

# 主循环
for fname in tqdm(os.listdir(input_dir)):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    in_path = os.path.join(input_dir, fname)
    out_path = os.path.join(output_dir, fname)

    try:
        img = Image.open(in_path)
        cropped = crop_largest_square_from_alpha(img)
        resized = cropped.resize(target_size, Image.BILINEAR)
        resized.save(out_path)
    except Exception as e:
        print(f"[ERROR] Failed to process {fname}: {e}")