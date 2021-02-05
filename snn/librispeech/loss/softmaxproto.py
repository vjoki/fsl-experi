# Adapted from https://github.com/clovaai/voxceleb_trainer/
import torch
import torch.nn as nn
from .softmax import SoftmaxLoss
from .angularproto import AngularPrototypicalLoss


class SoftmaxPrototypicalLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.softmax = SoftmaxLoss(**kwargs)
        self.angleproto = AngularPrototypicalLoss(**kwargs)

    def forward(self, x: torch.Tensor, label: torch.Tensor):
        nlossS, prec1 = self.softmax(x.reshape(-1, x.size(-1)), label.flatten())
        if torch.isnan(nlossS) or torch.isinf(nlossS):
            print('Softmax loss is NaN or inf', x, label)
        nlossP, _ = self.angleproto(x, None)
        return nlossS+nlossP, prec1
