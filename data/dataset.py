import nibabel as nib
import imageio

import numpy as np

from pathlib import Path
from fastcore.utils import *
from torch.utils.data.dataset import Dataset
import torch

class NiftiDataset(Dataset):
    def __init__(self, source_directory, target_directory, transforms=None):
        self.source_directory = sorted(Path(source_directory).ls(file_exts=['.gz']))
        self.target_directory = sorted(Path(target_directory).ls(file_exts=['.gz']))
        self.transforms = transforms

    def __len__(self):
        return len(self.source_directory)

    def __getitem__(self, idx):
        t1_image = nib.load(self.source_directory[idx])
        t1_image = t1_image.get_fdata()
        t2_image = nib.load(self.target_directory[idx])
        t2_image = t2_image.get_fdata()

        data = {'t1': t1_image, 't2': t2_image}

        if self.transforms:
            data = self.transforms(data)

        return data



