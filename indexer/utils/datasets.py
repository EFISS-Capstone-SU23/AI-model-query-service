import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DeepHashingDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def get_image(self, img_path: str):
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data[idx]
        try:
            img = self.get_image(filename)
        except Exception as e:
            print(f"Error when loading image: {filename}")
            print(e)
            return None
        return img
    
    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
