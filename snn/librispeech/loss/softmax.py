# Adapted from https://github.com/clovaai/voxceleb_trainer/
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet.utils import accuracy


class SoftmaxLoss(nn.Module):
    def __init__(self, nOut=512, nClasses=251, **kwargs):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.fc = nn.Linear(nOut, nClasses)

    def forward(self, x, label=None):
        # TODO: Not sure what the rationale for N -> nClasses is...
        # B*W*SxN -> B*W*SxnClasses
        x = F.normalize(x, p=2, dim=1)
        x = self.fc(x)
        nloss = self.criterion(x, label)
        prec1 = accuracy(x.detach(), label.detach(), topk=(1,))[0]
        return nloss, prec1
