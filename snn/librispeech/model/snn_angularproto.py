from typing import Tuple, List
import torch
import torch.nn.functional as F

from .base import BaseNet
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
        support_sets, _ = batch

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

        return loss

    def validation_step(self,  # type: ignore[override]
                        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        batch_idx: int):
        x1, x2, y = batch

        x1 = self.spectogram_transform(x1)
        x2 = self.spectogram_transform(x2)

        x1 = self.cnn(x1)
        x2 = self.cnn(x2)
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
        out = F.pairwise_distance(x1, x2, keepdim=True)

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

        x1 = self.cnn(x1)
        x2 = self.cnn(x2)
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
        out = F.pairwise_distance(x1, x2, keepdim=True)

        loss = F.binary_cross_entropy_with_logits(out, y)

        self.test_accuracy(out, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

        return [out, y]
