# %cd ../../..
# %pip install transformers
# %pip install datasets
from transformers import ViTImageProcessor, ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch
import torch.nn as nn
import faiss

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.classifier = nn.Identity()
import torch
import numpy as np
# from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
df = pd.read_csv('data/shopee_crop_yolo/database.txt', sep=' ', nrows=None, names=['img_path', 'label']).drop('label', axis=1)
df
from datasets import Dataset
dataset = Dataset.from_pandas(df)

chunk_size = 100000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)

def tokenize_function(examples):
    img = cv2.imread(examples["img_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].squeeze()
    return inputs

for chunk in tqdm(range(0, len(dataset), chunk_size)):
    print(f"Processing chunk {chunk}")
    tokenized_dataset = Dataset.from_dict(dataset[chunk:chunk+chunk_size]).map(tokenize_function, batched=False, batch_size=100, writer_batch_size=100)
    tokenized_dataset = tokenized_dataset.remove_columns(['img_path'])
    tokenized_dataset.save_to_disk(f"/media/saplab/MinhNVMe/relahash/temp/tokenized_dataset_{chunk}.hf")

    tokenized_dataset = tokenized_dataset.with_format("torch", device=device)
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=512, shuffle=False)

    embeddings = []
    with torch.no_grad():
        for i, x in enumerate(tqdm(dataloader)):
            logits = model(**x).logits
            embeddings.append(logits.cpu())
        embeddings = torch.cat(embeddings)

    print(embeddings.shape)

    torch.save({
        "embedding": embeddings
    }, '/media/saplab/MinhNVMe/relahash/temp/embeddings_{}.pt'.format(chunk))

    # break