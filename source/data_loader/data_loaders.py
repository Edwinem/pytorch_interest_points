from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset

from base import BaseDataLoader


class SampleDataLoader(BaseDataLoader):
    def __init__(
        self, batch_size,
        shuffle, validation_split,
        num_workers,pin_memory=False, dataset_args={}
    ):
        self.dataset = SampleDataSet(**dataset_args)
        super(SampleDataLoader, self).__init__(
            self.dataset, batch_size, shuffle,
            validation_split, num_workers,pin_memory)


class SampleDataSet(Dataset):
    def __init__(self, data_dir, training):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST(
            data_dir, train=training, download=True, transform=trsfm
        )

    def __getitem__(self, index):
        frame, label = self.dataset.__getitem__(index)
        data = {
            "frame": frame,
            "label": label,
        }
        return data

    def __len__(self):
        return len(self.dataset)
        