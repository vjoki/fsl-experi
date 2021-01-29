from typing import Tuple, Optional
import torch

from .snn_angularproto import SNNAngularProto
from snn.librispeech.loss.softmaxproto import SoftmaxPrototypicalLoss


class SNNSoftmaxProto(SNNAngularProto):
    def __init__(self, n_speakers: Optional[int] = None, **kwargs):
        super().__init__(n_speakers=n_speakers, **kwargs)
        self.training_loss_fn = SoftmaxPrototypicalLoss(
            nOut=self.cnn_out_dim,
            nClasses=n_speakers if n_speakers else 251
        )

    def training_step(self,  # type: ignore[override]
                      batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        support_sets, labels = batch
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

        loss, acc = self.training_loss_fn(support, labels)

        self.log('train_acc_step', acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss
