TOTAL_SHARD = 300
NUM_WORKER = 4

"""
This script uses YOLOv8 to offline crop images and wraps it into a HuggingFace IterableDataset
"""

import numpy as np
import urllib
import requests
import cv2
import pandas as pd
import os
from tqdm.auto import tqdm
from datasets import Dataset, IterableDataset
from ultralytics import YOLO
import torch
from transformers import ViTImageProcessor, ViTForImageClassification, ViTFeatureExtractor
import torch.nn as nn
import pickle

import datasets
datasets.disable_caching()

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

output_dir = 'data/product_images/'
yolo_model_path = 'yolov8n_12ep_24-7_32.5mAP.pt'

# Define your YOLOv8-related functions here
def initialize_yolov8_model():
    # download https://data.efiss.tech/efiss/yolov8n_12ep_24-7_32.5mAP.pt 
    # and put it in the current directory
    if not os.path.exists(yolo_model_path):
        print(f"Downloading YOLOv8 model to {yolo_model_path}")
        urllib.request.urlretrieve('https://data.efiss.tech/efiss/yolov8n_12ep_24-7_32.5mAP.pt', yolo_model_path)
    model = YOLO(yolo_model_path)
    model.to('cuda:0')
    return model

def crop_image_with_yolov8(model: YOLO, img: np.ndarray) -> list[np.ndarray]:
    # YOLOv8 cropping logic here
    result = model.predict(
        source=img,
        conf=0.2,
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

def read_img_from_network(image_path: str) -> np.ndarray:
    # assert image_path.startswith('https://'), f"Image path {image_path} is not a valid URL"
    prefix = 'https://data.efiss.tech/efiss/'
    image_path = prefix + image_path
    # image_path = 'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1920&q=80'
    req = urllib.request.urlopen(image_path)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

model = initialize_yolov8_model()
def crop_image_to_multiple_images(row) -> dict:
    image_path = row['img_path']
    img = read_img_from_network(row['img_path'])
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

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
ranking_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
ranking_model.classifier = nn.Identity()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ranking_model.eval()
ranking_model.to(device)

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

def _filter1(x):
    return len(x['cropped_img_paths']) > 0

def _filter2(x):
    return len(x['pixel_values']) > 0

def main(shard_id: int):
    to_be_index: list[str] = []
    file_path = 'database_info.txt'
    data_path = 'https://data.efiss.tech/efiss/queue/files_list_efiss.txt'
    if not os.path.exists(file_path):
        print(f"Downloading file list from {data_path}")
        urllib.request.urlretrieve(data_path, file_path)
    with open('database_info.txt', 'r') as f:
        for line in tqdm(f.readlines(), desc='Reading file list from local'):
            to_be_index.append(line.strip())
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(pd.DataFrame({'img_path': to_be_index}))
    dataset = dataset.shard(num_shards=TOTAL_SHARD, index=shard_id)
    # dataset = Dataset.from_dict(dataset[:34])
    total_len = len(dataset)
    dataset = dataset.to_iterable_dataset()
    cropped_images = dataset.map(crop_image_to_multiple_images, batched=False, remove_columns=['img_path'])
    # filter out empty images
    cropped_images = cropped_images.filter(_filter1)
            
    tokenized_images = cropped_images.map(tokenize_function, batched=True, batch_size=1, remove_columns=['cropped_images'])
    tokenized_images = tokenized_images.with_format("torch")
    # filter out empty images
    tokenized_images = tokenized_images.filter(_filter2)
    dataloader = torch.utils.data.DataLoader(tokenized_images, batch_size=16, num_workers=NUM_WORKER)
    out_cropped_img_paths: list[str] = []
    out_embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for i, row in enumerate(tqdm(dataloader, total=total_len, desc='Extracting embeddings')):
            logits = ranking_model(pixel_values=row['pixel_values'].to(device)).logits  # (batch_size, 768)

            out_cropped_img_paths.extend(row['cropped_img_paths'])  # list[str]
            out_embeddings.append(logits.cpu())
    embeddings = torch.cat(out_embeddings, dim=0)
    payload = {
        'shard_id': shard_id,
        'embeddings': embeddings.cpu().numpy(),
        'cropped_img_paths': out_cropped_img_paths
    }
    # prepare to send via http
    dumped_payload: bytes = pickle.dumps(payload)
    # send
    url = 'https://indexer.efiss.tech/upload'
    r = requests.post(url, files={'file': ('file.pkl', dumped_payload, 'application/octet-stream')})
    print(r.status_code, r.reason)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard_id', type=int, default=0, help='Shard ID', required=True)
    args = parser.parse_args()
    main(args.shard_id)