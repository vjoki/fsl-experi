from typing import Tuple
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .base import BaseNet
from snn.librispeech.loss.angularproto import AngularPrototypicalLoss


class SNNAngularProto(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_loss_fn = AngularPrototypicalLoss()

        self.val_accuracy = pl.metrics.Accuracy(compute_on_step=False, threshold=1.3)
        self.test_accuracy = pl.metrics.Accuracy(compute_on_step=False, threshold=1.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.cnn(x)

    def training_step(self,  # type: ignore[override]
                      batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        support_sets, _ = batch
        #print('support_sets', support_sets.shape)
        #print('labels', labels.shape)

        # BxWxSxWAVE -> BxSxWxWAVE
        #support_sets = support_sets.transpose(2,1)

        # Group ways with batch, so that support_sets gets pushed through the CNN in one go without extra looping.
        # BxWxSxWAVE -> B*W*SxWAVE
        support_sets = support_sets.reshape(-1,
                                            support_sets.size(-1))

        #print('spectro in', support_sets.shape)
        # B*W*SxWAVE -> B*W*Sx1xNMELSxSPEC
        x = self.spectogram_transform(support_sets, augment=self.specaugment)
        #print('ResNet (spectro)', x.shape)
        # B*W*Sx1xNMELSxSPEC -> B*W*SxN
        support = self.cnn(x)
        #print('ResNet (out)', support.shape)

        # Restore the original shape, by restoring the ways from batch dim.
        # B*W*SxN -> BxWxSxN
        support = support.reshape(self.train_batch_size,
                                  -1,
                                  self.num_shots,
                                  support.size(-1))
        assert support.size(1) == self.num_ways

        loss, acc = self.training_loss_fn(support)

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
        # 0 => match, so flip y.
        y = 1 - y

        # out = F.pairwise_distance(x1.unsqueeze(-1), x2.unsqueeze(-1).transpose(0, 2))
        # out = -1 * torch.mean(out, dim=1)
        # out = out.unsqueeze(-1)

        loss = F.binary_cross_entropy_with_logits(out, y)

        self.val_accuracy(out, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return [out.detach(), y.detach()]

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
        # 0 => match, so flip y.
        y = 1 - y

        # out = F.pairwise_distance(x1.unsqueeze(-1), x2.unsqueeze(-1).transpose(0, 2))
        # out = -1 * torch.mean(out, dim=1)
        # out = out.unsqueeze(-1)

        loss = F.binary_cross_entropy_with_logits(out, y)

        self.test_accuracy(out, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

        return [out.detach(), y.detach()]
