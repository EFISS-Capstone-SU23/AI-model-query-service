"""
This scripts use YOLOv8 to offline cropping the images in the database to create new dataset
"""

import pandas as pd
import dask.dataframe as dd
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from tqdm.auto import tqdm
import os

model = YOLO('torchscripts_models/yolo/yolov8n_12ep_24-7_32.5mAP.pt')
model.to('cuda:0')
tqdm.pandas()

df = pd.read_csv('data/shopee/shopee-data-for-training-filter-corrupted.csv')

def crop_image(img: np.ndarray) -> list[np.ndarray]:
    """
    Crop the image using YOLOv8
    
    Args:
        img (np.ndarray): the image to crop
    
    Returns:
        np.ndarray: the cropped image, or leave the image unchanged if no object is detected
    """
    result = model.predict(
        source=img,
        conf=0.05,
        device='0',
        save=False,
        verbose=False
    )[0]
    if len(result.boxes.xyxy) == 0:
        print("No object detected")
        return []
    
    out = []
    for box in result.boxes.xyxy:
        x, y, _x, _y = list(box.int())
        out.append(result.orig_img[y:_y, x:_x])
    
    return out

### Begin
_df = df
num_threads = 10
ddf = dd.from_pandas(_df, npartitions=num_threads)

pbar = tqdm(total=len(df))

output_datadir = "data/shopee_crop_yolo/images"
def crop_image_to_multiple_images(image_path: str) -> list[str]:
    """
    Crop an image with YOLOv5 into multiple images. Return the paths of the cropped images.
    """
    img: np.ndarray = cv2.imread(image_path)
    # img_extension: str = image_path.split(".")[-1]
    img_name: str = image_path.split("/")[-1].split(".")[0]

    try:
        cropped_images: list[np.ndarray] = crop_image(img)
    except Exception as e:
        print(f"Error cropping image {image_path}: {e}")

    cropped_image_paths: list[str] = []
    for i, cropped_image in enumerate(cropped_images):
        cropped_image_path: str = f"{output_datadir}/{img_name}_crop{i}.jpg"
        cv2.imwrite(cropped_image_path, cropped_image)
        cropped_image_paths.append(cropped_image_path)

    pbar.update(1)
    return cropped_image_paths

def df_crop_image(df: pd.DataFrame) -> pd.DataFrame:
    df['cropped_img_paths'] = df['images'].apply(crop_image_to_multiple_images)
    pbar.update(1)
    return df

# https://stackoverflow.com/questions/45545110/make-pandas-dataframe-apply-use-all-cores

# with pbar:
#     prep = ddf.map_partitions(df_crop_image, meta={'images': 'str', 'class_idx': np.int64, 'cropped_img_paths': 'str'})
#     res: pd.DataFrame = prep.compute(scheduler='threads')
#     res.to_csv("data/shopee_crop_yolo/cropped_images.csv", index=False)

df['cropped_img_paths'] = df['images'].progress_apply(crop_image_to_multiple_images)
df.to_csv('data/shopee_crop_yolo/cropped_dataset.csv', index=False)
