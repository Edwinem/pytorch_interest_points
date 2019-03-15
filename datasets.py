from torch.utils.data import Dataset, DataLoader

import os

class CocoDataset(DataLoader):

    def __init__(self,dataset_path):
        self.path=os.path.join(dataset_path,'train2014')
        self.image_files=os.listdir(self.path)
