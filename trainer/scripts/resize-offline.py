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

SIZE = 600

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

with open('database_info.txt', 'r') as f:
    lines = f.readlines()
    img_paths = [line.strip() for line in lines]

corrupted_images = []
resized_img_paths = []

print("Begin resizing images...")
for file_path in tqdm(img_paths):
    _file_path = rename_path_to_original_extension(file_path)
    if _file_path is None:
        print(f"Corrupted image: {file_path}")
        corrupted_images.append(file_path)
        continue
    elif file_path != _file_path:
        print(f"Renamed image from {file_path} to {_file_path}")
    file_path = _file_path
    abs_path = os.path.abspath(file_path)
    if file_path.endswith('.jpeg'):
        with open(abs_path, 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            print(f"Not complete image: {abs_path}")
            corrupted_images.append(file_path)
            continue
    
    img = cv2.imread(abs_path)
    img = cv2.resize(img, (SIZE, SIZE))
    new_path = file_path.replace('output/', f'resize_{SIZE}x{SIZE}/')
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    result = cv2.imwrite(new_path, img)
    if not result:
        print('Error: ', file_path)
        corrupted_images.append(file_path)
    else:
        resized_img_paths.append(new_path)

print("Done resizing images!")
print("coprrupted images: ", corrupted_images)

with open(f'resized_{SIZE}x{SIZE}_database_info.txt', 'w') as f:
    f.write('\n'.join(resized_img_paths))

with open('corrupted_images.txt', 'w') as f:
    f.write('\n'.join(corrupted_images))