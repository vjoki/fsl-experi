from typing import Tuple
import torch
import torch.nn.functional as F

from snn.librispeech.model.base import BaseNet


class SNN(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x1 = self.cnn(x1)
        x2 = self.cnn(x2)
        dist = torch.abs(x1 - x2)
        return self.out(dist)

    def training_step(self,  # type: ignore[override]
                      batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x1, x2, y = batch

        x1 = self.spectogram_transform(x1, augment=self.specaugment)
        x2 = self.spectogram_transform(x2, augment=self.specaugment)
        out = self(x1, x2)

        # dist = F.pairwise_distance(x1, x2, keepdim=True)
        # loss = torch.mean((1.0 - y) * torch.pow(dist, 2) + y * torch.pow(torch.clamp(1.0 - dist, min=0.0), 2))
        loss = F.binary_cross_entropy_with_logits(out, y)

        acc = self.train_accuracy(out, y)
        self.log('train_acc_step', acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_label_avg', y.mean(), on_step=True, on_epoch=True)

        return loss

    def validation_step(self,  # type: ignore[override]
                        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        batch_idx: int):
        x1, x2, y = batch

        x1 = self.spectogram_transform(x1)
        x2 = self.spectogram_transform(x2)
        out = self(x1, x2)
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
        out = self(x1, x2)
        loss = F.binary_cross_entropy_with_logits(out, y)

        self.test_accuracy(out, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

        return [out, y]
