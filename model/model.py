import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch


class SuperPointNiko(BaseModel):
    def __init__(self,use_batch_norm=False):
        super(SuperPointNiko, self).__init__()


        self.use_batch_norm=use_batch_norm




        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        #Encoder
        self.conv1a = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # Detector portion


        self.convPa = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        # descriptor portion
        self.convDa = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        if(self.use_batch_norm):
            self.b_1a=nn.BatchNorm2d(64)
            self.b_1b = nn.BatchNorm2d(64)
            self.b_2a = nn.BatchNorm2d(64)
            self.b_2b = nn.BatchNorm2d(64)
            self.b_3a = nn.BatchNorm2d(128)
            self.b_3b = nn.BatchNorm2d(128)
            self.b_4a = nn.BatchNorm2d(128)
            self.b_4b = nn.BatchNorm2d(128)





    def forward(self, x):

        if self.use_batch_norm:
            #1
            x=self.conv1a(x)
            x=self.relu(x)
            x=self.b_1a(x)

            x=self.conv1b(x)
            x=self.relu(x)
            x=self.b_1b(x)
            x=self.pool(x)

            #2
            x = self.conv2a(x)
            x = self.relu(x)
            x = self.b_2a(x)

            x = self.conv2b(x)
            x = self.relu(x)
            x = self.b_2b(x)
            x=self.pool(x)

            #3
            x = self.conv3a(x)
            x = self.relu(x)
            x = self.b_3a(x)

            x = self.conv3b(x)
            x = self.relu(x)
            x = self.b_3b(x)
            x=self.pool(x)

            #4

            x = self.conv4a(x)
            x = self.relu(x)
            x = self.b_4a(x)

            x = self.conv4b(x)
            x = self.relu(x)
            x = self.b_4b(x)
            x=self.pool(x)
        else:
            # 1
            x = self.conv1a(x)
            x = self.relu(x)

            x = self.conv1b(x)
            x = self.relu(x)
            x = self.pool(x)

            # 2
            x = self.conv2a(x)
            x = self.relu(x)

            x = self.conv2b(x)
            x = self.relu(x)
            x = self.pool(x)

            # 3
            x = self.conv3a(x)
            x = self.relu(x)

            x = self.conv3b(x)
            x = self.relu(x)
            x = self.pool(x)

            # 4

            x = self.conv4a(x)
            x = self.relu(x)

            x = self.conv4b(x)
            x = self.relu(x)
            x = self.pool(x)

        #Detector
        det=self.relu(self.convPa(x))
        heatmap=self.convPb(det)

        #Desc
        desc=self.relu(self.convDa(x))
        desc=self.convDb(x)

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))
        return heatmap, desc