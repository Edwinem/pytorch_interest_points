from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        return self.loss_fn(logits, target)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss_fn = nn.MSELoss

    def forward(self, logits, target):
        return self.loss_fn(logits, target)
