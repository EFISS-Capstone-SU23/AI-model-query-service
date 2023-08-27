"""
This script uses YOLOv8 to offline crop images and wraps it into a HuggingFace IterableDataset
"""

import numpy as np
import cv2
import pandas as pd
import os
from tqdm.auto import tqdm
from datasets import Dataset, IterableDataset
from ultralytics import YOLO
from google.cloud import storage
import torch
from transformers import ViTImageProcessor, ViTForImageClassification, ViTFeatureExtractor
import torch.nn as nn

import datasets
datasets.disable_caching()

# from torch.multiprocessing import Pool, Process, set_start_method
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass

TOTAL_SHARD = 24
SHARD_ID = 0

# Define your YOLOv8-related functions here
def initialize_yolov8_model():
    model = YOLO('torchscripts_models/yolo/yolov8n_12ep_24-7_32.5mAP.pt')
    model.to('cuda:0')
    return model

def crop_image_with_yolov8(model, img):
    # YOLOv8 cropping logic here
    result = model.predict(
        source=img,
        conf=0.3,
        device='0',
        save=False,
        verbose=False
    )[0]
    # Crop and return images
    cropped_images = []
    for box in result.boxes.xyxy:
        x, y, _x, _y = list(box.int())
        cropped_images.append(result.orig_img[y:_y, x:_x])
    return cropped_images

output_dir = 'data/product_images/'

# Set up GCS client
client = storage.Client()
bucket_name = 'efiss'
bucket = client.get_bucket(bucket_name)
def read_img_from_GCS(image_path) -> np.ndarray:
    # Get blob from GCS
    blob = bucket.blob(image_path)

    # Read image from blob
    print(f"Dowloading {image_path} from GCS")
    img_bytes = blob.download_as_bytes()
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    print(f"Downloaded {image_path} from GCS")

    return img
model = initialize_yolov8_model()
def crop_image_to_multiple_images(row) -> dict:
    image_path = row['img_path']
    img = read_img_from_GCS(row['img_path'])
    img_name = image_path.split("/")[-1].split(".")[0]

    try:
        cropped_images = crop_image_with_yolov8(model, img)
    except Exception as e:
        print(f"Error cropping image {image_path}: {e}")
        cropped_images = []

    cropped_image_paths = []
    for i, cropped_image in enumerate(cropped_images):
        cropped_image_path = os.path.join(output_dir, f"{img_name}_crop{i}.jpg")
        # cv2.imwrite(cropped_image_path, cropped_image)
        cropped_image_paths.append(cropped_image_path)

    # return {'cropped_img_paths': cropped_image_paths, 'cropped_images': cropped_images}
    print(f"Finished cropping {image_path}: {len(cropped_images)} images")
    row['cropped_img_paths'] = cropped_image_paths
    row['cropped_images'] = cropped_images
    return row

# row.keys()
# row['cropped_img_paths']
import matplotlib.pyplot as plt
def plot_img(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
# plot_img(row['cropped_images'][0])
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
ranking_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
ranking_model.classifier = nn.Identity()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ranking_model.eval()
ranking_model.to(device)
...

to_be_index: list[str] = []
with open('to_be_index_efiss.txt', 'r') as f:
    for line in f.readlines():
        to_be_index.append(line.strip())

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(pd.DataFrame({'img_path': to_be_index}))
dataset
dataset = dataset.shard(num_shards=TOTAL_SHARD, index=SHARD_ID)
# dataset = Dataset.from_dict(dataset[:34])
dataset
total_len = len(dataset)
# dataset = dataset.to_iterable_dataset()
cropped_images = dataset.map(crop_image_to_multiple_images, batched=False, remove_columns=['img_path'])
cropped_images
# tokenize
def tokenize_function(row):
    imgs: list[np.ndarray] = row["cropped_images"][0]  # batch size 1
    out = None
    for img in imgs:
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs = processor(images=img, return_tensors="pt")
        inputs['pixel_values'] = [inputs['pixel_values'].squeeze()]
        if not out:
            out = inputs
        else:
            out['pixel_values'].append(inputs['pixel_values'][0])
    if out:
        out['cropped_img_paths'] = row['cropped_img_paths'][0] or []
        return out
    else:
        return {'pixel_values': [], 'cropped_img_paths': []}
        
tokenized_images = cropped_images.map(tokenize_function, batched=True, batch_size=1, remove_columns=['cropped_images'])
tokenized_images.set_format("torch", device=device)
dataloader = torch.utils.data.DataLoader(tokenized_images, batch_size=16, num_workers=0)
out = []
for i, row in enumerate(tqdm(dataloader, total=total_len)):
    # print(row)
    out.append(row)
    if i == 10:
        break
# TODO: insert to milvus
out[0].keys()
