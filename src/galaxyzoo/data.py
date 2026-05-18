import os
from pathlib import Path
from typing import Literal

import numpy as np
from numpy import ndarray, int64
import pandas

import torch
from torch.utils.data import Dataset
from PIL import Image

def load_data(img_path: Path, labels_path: Path, task: Literal["task1", "task2", "task3"]):
    if task == "task1":
        #obtaining only the relevant labels
        raw_labels = pandas.read_csv(labels_path)
        labels = raw_labels.iloc[:,1:4]
        labels_numpy = labels.to_numpy()
        labels_class = labels_numpy.argmax(axis=1) #we want the most voted answer

    #obtaining the set of images
    image_paths = [
        os.path.join(img_path, str(name)+'.jpg')
        for name in raw_labels['GalaxyID']
    ]

    return image_paths, labels_class

class GalaxiesDataset(Dataset):
    def __init__(self, image_paths: Path, labels: ndarray, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label