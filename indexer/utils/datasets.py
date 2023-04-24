import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets.folder import pil_loader, accimage_loader

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DeepHashingDataset(Dataset):
    def __init__(self, data, transform=None):
        if torchvision.get_image_backend() == "PIL":
            self.loader = pil_loader
        else:
            self.loader = accimage_loader

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data[idx]
        img = self.loader(filename)
        if self.transform:
            img = self.transform(img)
        return img
