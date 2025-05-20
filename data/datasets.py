import os
import glob
import torch
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

class CIFAR10Dataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted(self.data.iloc[:, 1].unique().tolist())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.data.iloc[idx, 0]))
        image = Image.open(f"{img_name}.png")
        label = self.data.iloc[idx, 1]
        label = torch.tensor(self.class_to_idx[label])

        if self.transform:
            image = self.transform(image)

        return image, label
    
class STL10Dataset(Dataset):

    def __init__(self, json_file, root_dir, transform=None):
        self.data = pd.read_json(json_file)
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted(self.data.iloc[:, 1].unique().tolist())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.data.iloc[idx, 0]).split('/')[-1])
        image = Image.open(img_name)
        label = self.data.iloc[idx, 1]
        label = torch.tensor(self.class_to_idx[label])

        if self.transform:
            image = self.transform(image)

        return image, label
    
class STL10UnsupervisedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = sorted(glob.glob(f'{root_dir}/*.png'))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_name = self.files[idx]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image
