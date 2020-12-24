from typing import Tuple, List, Any
from typing_extensions import Final
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNet
from capsnet.CapsNet import CapsNetWithoutPrimaryCaps


class SNNCapsNet(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 2xC -> 4x128
        self.caps: Final[nn.Module] = CapsNetWithoutPrimaryCaps(
            routing_iterations=3,
            input_caps=2, input_dim=self.cnn_out_dim,
            output_caps=4, output_dim=128
        )
        caps_out_dim: Final = 4*128

        # (4*128) -> 1
        self.out: nn.Module = nn.Linear(caps_out_dim, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x1 = self.cnn(x1)
        x2 = self.cnn(x2)

        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
        x = torch.stack((x1, x2), dim=1)
        x, _ = self.caps(x)
        x = torch.flatten(x, 1)
        return self.out(x)

    def training_step(self,  # type: ignore[override]
                      batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x1, x2, y = batch

        x1 = self.spectogram_transform(x1, augment=self.specaugment)
        x2 = self.spectogram_transform(x2, augment=self.specaugment)
        out = self(x1, x2)

        loss = F.binary_cross_entropy_with_logits(out, y)

        acc = self.train_accuracy(out, y)
        self.log('train_acc_step', acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_label_avg', y.mean(), on_step=True, on_epoch=True)

        return loss

    def training_epoch_end(self, training_step_outputs: List[Any]):
        self.log('train_acc_epoch', self.train_accuracy.compute())

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
