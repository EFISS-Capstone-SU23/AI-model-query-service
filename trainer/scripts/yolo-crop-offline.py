"""
This scripts use YOLOv8 to offline cropping the images in the database, then resize it
"""

from typing import Optional
import shutil
import os
import sys
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import albumentations as A
import pandas as pd
import json
from ultralytics import YOLO
import cv2

SIZE = 300

model = YOLO('torchscripts_models/yolo/yolov8n_12ep_640x_23-7.pt')
model.to('cuda:0')

outliers = []

def crop_image(img: np.ndarray) -> np.ndarray:
    """
    Crop the image using YOLOv8
    
    Args:
        img (np.ndarray): the image to crop
    
    Returns:
        np.ndarray: the cropped image, or leave the image unchanged if no object is detected
    """
    result = model.predict(
        source=img,
        conf=0.03,  # set low-confidence threshold instead of 0.25
        save=False,
        verbose=False
    )[0]
    if len(result.boxes.xyxy) == 0:
        return None
    x, y, _x, _y = list(result.boxes.xyxy[0].int())
    return result.orig_img[y:_y, x:_x]

def rename_path_to_original_extension(img_path: str) -> Optional[str]:
    try:
        with Image.open(img_path) as img:
            img.verify()
            image_format = img.format.lower()
            if image_format == 'jpeg' and not (img_path.endswith('.jpg') or img_path.endswith('.jpeg')):
                new_filename = img_path.replace('.jpeg', '.jpg')
                new_filename = img_path.replace('.jpg', f'.{image_format}')
                # os.rename(img_path, new_filename)
                shutil.move(img_path, new_filename)
                return new_filename
            else:
                return img_path
    except Exception as e:
        print(e)
        return None

# for now, we will read from the resized database_info.txt
with open('database_info.txt', 'r') as f:
    lines = f.readlines()
    img_paths = [line.strip() for line in lines]

corrupted_images = []
resized_img_paths = []

print("Begin resizing images...")
for file_path in tqdm(img_paths):
    file_path = file_path.replace(f'yolo_resize_{SIZE}x{SIZE}/', 'resize_600x600/')
    __new_path = file_path.replace('resize_600x600/', f'yolo_resize_{SIZE}x{SIZE}/')
    if os.path.exists(__new_path):
        resized_img_paths.append(__new_path)
        print(f"Skip resized image: {file_path}")
        continue

    # _file_path = rename_path_to_original_extension(file_path)
    _file_path = file_path
    if _file_path is None:
        print(f"Corrupted image: {file_path}")
        corrupted_images.append(file_path)
        continue
    elif file_path != _file_path:
        print(f"Renamed image from {file_path} to {_file_path}")
    file_path = _file_path
    abs_path = os.path.abspath(file_path)
    try:
        # if file_path.endswith('.jpeg'):
        #     with open(abs_path, 'rb') as f:
        #         check_chars = f.read()[-2:]
        #     if check_chars != b'\xff\xd9':
        #         print(f"Not complete image: {abs_path}")
        #         corrupted_images.append(file_path)
        #         continue
        
        img = cv2.imread(abs_path)
        _img = crop_image(img)
        if _img is None:
            print(f"No object detected: {abs_path}")
            _img = img
            outliers.append(file_path)
        img = cv2.resize(_img, (SIZE, SIZE))
        new_path = file_path.replace('resize_600x600/', f'yolo_resize_{SIZE}x{SIZE}/')
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        if os.path.exists(new_path):
            raise Exception(f"File already exists: {new_path}")
        result = cv2.imwrite(new_path, img)
        if not result:
            print('Error: ', file_path)
            corrupted_images.append(file_path)
        else:
            resized_img_paths.append(new_path)
    except Exception as e:
        if "File already exists" in str(e):
            raise e
        print(e)
        print('Error: ', file_path)
        corrupted_images.append(file_path)
        continue

print("Done resizing images!")
print("coprrupted images: ", corrupted_images)

with open(f'yolo_resized_{SIZE}x{SIZE}_database_info.txt', 'w') as f:
    f.write('\n'.join(resized_img_paths))

with open('corrupted_images.txt', 'w') as f:
    f.write('\n'.join(corrupted_images))

with open('outliers.txt', 'w') as f:
    f.write('\n'.join(outliers))