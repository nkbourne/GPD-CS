import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from sources.data.base import GrayImagePaths


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example

class GCustomTrain(CustomBase):
    def __init__(self, size, crop_size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = GrayImagePaths(paths=paths, size=size, crop_size=crop_size, random_crop=True)


class GCustomTest(CustomBase):
    def __init__(self, size, crop_size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = GrayImagePaths(paths=paths, size=size, crop_size=crop_size, random_crop=True)

