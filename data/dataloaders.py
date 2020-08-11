from torch.utils.data import DataLoader
from .transforms import ToTensor
from .dataset import NiftiDataset


def get_dataloader(source_directory, target_directory, transforms=ToTensor, batch_size=8, num_workers=8):
    nifti_small_dataset = NiftiDataset(source_directory=source_directory, target_directory=target_directory, transforms=transforms)
    return DataLoader(nifti_small_dataset, batch_size=batch_size, num_workers=num_workers)