from typing import Tuple, List
import torch
import torch.nn.functional as F

from snn.librispeech.model.base import BaseNet
from snn.librispeech.loss.angularproto import AngularPrototypicalLoss
from resnet.utils import accuracy


class SNNAngularProto(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_loss_fn = AngularPrototypicalLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.cnn(x)

    def training_step(self,  # type: ignore[override]
                      batch: Tuple[List[List[torch.Tensor]], torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        support_sets, labels = batch

        supports = []
        for shots in support_sets:
            s = []
            for waveform in shots:
                x = self.spectogram_transform(waveform, augment=self.specaugment)
                s.append(self.cnn(x))
            supports.append(torch.cat(s, dim=0))

        support = torch.stack(supports, dim=0)

        loss, cos_sim_matrix, label = self.training_loss_fn(support)
        acc = accuracy(cos_sim_matrix.detach(), label.detach(), topk=(1,))[0]

        # acc = self.train_accuracy(out, y)
        self.log('train_acc_step', acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log('train_label_avg', y.mean(), on_step=True, on_epoch=True)

        return loss

    def validation_step(self,  # type: ignore[override]
                        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        batch_idx: int):
        x1, x2, y = batch

        x1 = self.spectogram_transform(x1)
        x2 = self.spectogram_transform(x2)
        #out = self(x1, x2)

        x1 = self.cnn(x1)
        x2 = self.cnn(x2)
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
        out = F.pairwise_distance(x1, x2, keepdim=True)
        #out = torch.mean(dist)
        #out = 1 - F.normalize(dist, p=2, dim=1)

        # x1 = self.cnn(x1)
        # x2 = self.cnn(x2)
        # x2 = torch.stack([x2], dim=1).unsqueeze(2)
        # long_y = y.squeeze(1).long()
        # loss, out, label = self.loss_fn(x1, x2, long_y)
        # acc = accuracy(out, label, topk=(1,))[0]
        # out = torch.mean(out, 1, keepdim=True)
        # self.log('val_acc', acc, on_step=True, on_epoch=True)

        ## dist = F.pairwise_distance(x1, x2, keepdim=True)
        #loss = torch.mean((1.0 - y) * torch.pow(out, 2) + y * torch.pow(torch.clamp(1.0 - out, min=0.0), 2))
        #loss = self.loss(out, y)
        loss = F.binary_cross_entropy_with_logits(out, y)

        self.val_accuracy(out, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return [out, y]

    def test_step(self,  # type: ignore[override]
                  batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                  batch_idx: int):
        x1, x2, y = batch

        x1 = self.spectogram_transform(x1)
        x2 = self.spectogram_transform(x2)
        #out = self(x1, x2)

        x1 = self.cnn(x1)
        x2 = self.cnn(x2)
        #print('x', x1, x2)
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
        out = F.pairwise_distance(x1, x2, keepdim=True)
        #print('out', out)
        #print('y', y)
        #out = torch.mean(dist)
        #out = 1 - F.normalize(dist, p=2, dim=1)

        # x1 = self.cnn(x1)
        # x2 = self.cnn(x2)
        # x2 = torch.stack([x2], dim=1).unsqueeze(2)
        # long_y = y.squeeze(1).long()
        # loss, out, label = self.loss_fn(x1, x2, long_y)
        # acc = accuracy(out, label, topk=(1,))[0]
        # out = torch.mean(out, 1, keepdim=True)
        # self.log('test_acc_step', acc, on_step=True, on_epoch=False)

        # dist = F.pairwise_distance(x1, x2, keepdim=True)
        #loss = torch.mean((1.0 - y) * torch.pow(dist, 2) + y * torch.pow(torch.clamp(1.0 - dist, min=0.0), 2))
        #loss = torch.mean((1.0 - y) * torch.pow(out, 2) + y * torch.pow(torch.clamp(1.0 - out, min=0.0), 2))
        #loss = self.loss(out, y)
        loss = F.binary_cross_entropy_with_logits(out, y)

        self.test_accuracy(out, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

        return [out, y]
