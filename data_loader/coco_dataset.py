from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset


import albumentations as A

from base import BaseDataLoader


class COCODataLoader(BaseDataLoader):
    def __init__(
        self, batch_size,
        shuffle, validation_split,
        num_workers,pin_memory=False, dataset_args={}
    ):
        self.dataset = COCODataSet(**dataset_args)
        super(COCODataLoader, self).__init__(
            self.dataset, batch_size, shuffle,
            validation_split, num_workers,pin_memory)


class COCODataSet(Dataset):
    def __init__(self, data_dir, training):


    def __getitem__(self, index):
        frame, label = self.dataset.__getitem__(index)
        data = {
            "frame": frame,
            "label": label,
        }
        return data

    def __len__(self):
        return len(self.dataset)