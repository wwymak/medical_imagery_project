import numpy as np
from fastcore.utils import *
import torch

class RandomCrop3D:
    """Crop randomly the 3d image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        t1, t2 = sample['t1'], sample['t2']

        height, width, depth = t1.shape[:3]
        new_h, new_w, new_depth = self.output_size

        top = np.random.randint(0, height - new_h)
        left = np.random.randint(0, width - new_w)
        bottom = np.random.randint(0, depth - new_depth)

        t1_cropped = t1[top: top + new_h, left: left + new_w, bottom: bottom + new_depth]
        t2_cropped = t2[top: top + new_h, left: left + new_w, bottom: bottom + new_depth]

        return {'t1': t1_cropped, 't2': t2_cropped}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        t1, t2 = sample['t1'], sample['t2']
        t1 = t1[np.newaxis, :]
        t2 = t2[np.newaxis, :]
        return {'t1': torch.from_numpy(t1),
                't2': torch.from_numpy(t2)}