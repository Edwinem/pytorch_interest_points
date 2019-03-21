from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset



from base import BaseDataLoader


import os


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
    default_config = {
        'labels': None,
        'cache_in_memory': False,
        'validation_size': 100,
        'truncate': None,
        'preprocessing': {
            'resize': [240, 320]
        },
        'num_parallel_calls': 10,
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        },
        'warped_pair': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
    }



    def __init__(self, data_dir, training,config=default_config):
        basepath=os.path.join


        print('')

    def __getitem__(self, index):
        frame, label = self.dataset.__getitem__(index)
        data = {
            "frame": frame,
            "label": label,
        }
        return data

    def __len__(self):
        return len(self.dataset)