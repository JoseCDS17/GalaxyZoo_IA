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
    raw_labels = pandas.read_csv(labels_path)
    if task == "task1":
        #obtaining only the relevant labels
        labels = raw_labels[['Class1.1','Class1.2', 'Class1.3']]
        labels_numpy = labels.to_numpy()
        labels_out = labels_numpy.argmax(axis=1) #we want the most voted answer
    elif task == "task2":
        #obtaining only the relevant labels
        labels = raw_labels[['Class2.1','Class2.2', 'Class7.1','Class7.2','Class7.3']]
        labels_out = labels.to_numpy()
    elif task == "task3":
        #obtaining only the relevant labels
        labels = raw_labels[['Class2.1','Class2.2','Class6.1','Class6.2', 
                             'Class7.1','Class7.2','Class7.3',
                             'Class8.1','Class8.2','Class8.3','Class8.4','Class8.5','Class8.6','Class8.7']]
        labels_out = labels.to_numpy()

    #obtaining the set of images
    image_paths = [
        os.path.join(img_path, str(name)+'.jpg')
        for name in raw_labels['GalaxyID']
    ]

    return image_paths, labels_out

class GalaxiesDataset(Dataset):
    def __init__(self, image_paths: Path, labels: ndarray, transform=None, 
                 task: Literal["classification", "regression"] = "classification"):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.task = task

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        
        if self.transform:
            img = self.transform(img)

        if self.task == "classification":
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return img, label