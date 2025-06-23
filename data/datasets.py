import os
import glob
import torch
import pandas as pd

from PIL import Image
from typing import List, Optional, Dict, Callable
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
        img_name = os.path.join(
            self.root_dir, str(self.data.iloc[idx, 0]).split("/")[-1]
        )
        image = Image.open(img_name)
        label = self.data.iloc[idx, 1]
        label = torch.tensor(self.class_to_idx[label])

        if self.transform:
            image = self.transform(image)

        return image, label


class STL10UnsupervisedDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = sorted(glob.glob(f"{root_dir}/*.png"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image


class STL10DINODataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transforms: Optional[Dict[str, Callable]] = None,
        num_all_views: Optional[int] = None,
        num_global_views: Optional[int] = None,
    ):
        self.root_dir = root_dir
        self.transforms = transforms
        self.files = sorted(glob.glob(f"{root_dir}/*.png"))
        self.num_all_views = num_all_views
        self._num_global_views = num_global_views

    @property
    def num_global_views(self) -> int:
        return self._num_global_views

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        image = Image.open(img_name)
        image_views = self._get_dino_views(image)

        return image_views

    def _get_dino_views(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        From the already transformed image, creates N views, M global views(area > image_area * 50%) and N-M local views (area < image_area * 50%)
        """
        global_views = self._get_global_views(image.copy())
        local_views = self._get_local_views(image.copy())
        global_views.extend(local_views)
        return global_views

    def _get_global_views(self, image: torch.Tensor) -> List[torch.Tensor]:
        return [self.transforms["globals"](image) for _ in range(self.num_global_views)]

    def _get_local_views(self, image: torch.Tensor) -> List[torch.Tensor]:
        num_local_views = self.num_all_views - self.num_global_views
        return [self.transforms["locals"](image) for _ in range(num_local_views)]
