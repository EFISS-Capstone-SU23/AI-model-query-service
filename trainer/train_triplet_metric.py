"""
This file contains the code for training a simple triplet loss metric learning model.
This model is used to avoid collisions of the deep hashing model by reranking
collisioned sample.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import ast
import random

# Define hyperparameters and other configurations
margin = 1.0
batch_size = 128
num_epochs = 24
lr = 0.0005

class ShopeeDataset(Dataset):
    def __init__(self,
            filename: str,
            transform=None,
            target_transform=None,
        ) -> None:
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.df = pd.read_csv(filename, sep=' ', header=None, names=['file_path', 'label'])
        self.label_to_idxs = {}
        for idx, row in self.df.iterrows():
            label = row['label']
            if label not in self.label_to_idxs:
                self.label_to_idxs[label] = []
            self.label_to_idxs[label].append(idx)
        self.all_files = self.df['file_path'].values
    
    def get_image(self, img_path: str):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)['image'].float()
        return img

    def __getitem__(self, index):
        row = self.df.iloc[index]
    
        label = row['label']
        
        positive_anchor: str = row['file_path']
        positive_sample: str = self.df.iloc[random.choice(self.label_to_idxs[label])]['file_path']
        negative_sample: str = random.choice(self.all_files)

        positive_anchor = self.get_image(positive_anchor)
        positive_image = self.get_image(positive_sample)
        negative_image = self.get_image(negative_sample)

        return positive_anchor, positive_image, negative_image

    def __len__(self):
        return len(self.df.index)

dataset = ShopeeDataset('./data/shopee/database.txt', transform=A.Compose([
    A.Resize(500, 500),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightness(limit=0.2, p=0.75),
    A.RandomContrast(limit=0.2, p=0.75),
    # A.OneOf([
    #     A.OpticalDistortion(distort_limit=1.),
    #     A.GridDistortion(num_steps=5, distort_limit=1.),
    # ], p=0.75),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=40, val_shift_limit=0, p=0.75),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=30, border_mode=0, p=0.75),
    A.Blur(blur_limit=3, p=0.3),

    # A.RandomCrop(300, 300),
    A.RandomResizedCrop(224, 224, ratio=(0.2, 3)),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2()
]))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = nn.PairwiseDistance()
    
    def forward(self, anchor, positive, negative):
        distance_pos = self.distance(anchor, positive)
        distance_neg = self.distance(anchor, negative)
        loss = torch.relu(distance_pos - distance_neg + self.margin)
        return loss.mean()

import timm
class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        # self.model = timm.create_model('tf_efficientnetv2_b3', pretrained=True, num_classes=0)  # 1536
        self.model = timm.create_model('mobilenetv3_small_050', pretrained=True, num_classes=0)  # 1024
        # self.fc = nn.Linear(1536, 128)
        self.fc = nn.Linear(1024, 128)

    def forward(self, input):
        input = self.model(input)
        input = self.fc(input)
        return input



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# Create the model, loss function, and optimizer
model = TripletNet()
model.to(device)
criterion = TripletLoss(margin)
optimizer = optim.Adam(model.parameters(), lr=lr)
from tqdm import tqdm

# Training loop
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    bar = tqdm(dataloader, total=len(dataloader), position=0, leave=True, desc=f'Epoch {epoch+1}/{num_epochs}', ncols=100)
    for batch in bar:
        anchor, positive, negative = batch
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)
        
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        bar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")
# save model
torch.save(model.state_dict(), './metric-learning-model-small.pth')
