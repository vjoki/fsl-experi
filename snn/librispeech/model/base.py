import platform
from typing import Optional, List
import argparse
from typing_extensions import Final
import torch
import torch.nn as nn
import torchaudio
import pytorch_lightning as pl

# from pytorch_metric_learning.losses import ContrastiveLoss
# from pytorch_metric_learning.distances import LpDistance
from resnet.ResNetSE34V2 import MainModel as ThinResNet
from resnet.ResNetSE34L import MainModel as FastResNet
from resnet.utils import PreEmphasis
from snn.librispeech.utils import compute_evaluation_metrics

if platform.system().lower().startswith('win'):
    torchaudio.set_audio_backend("soundfile")
    torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
else:
    torchaudio.set_audio_backend("sox_io")


class BaseNet(pl.LightningModule):
    # NOTE: Defaults here shouldn't really matter much, they're just here to make initializing the model
    # for other purposes easier (such as log_graph)...
    def __init__(self, model: str,
                 learning_rate: float = 1e-3, max_epochs: int = 100,
                 augment: bool = False,
                 specaugment: bool = False,
                 n_fft: int = 512,
                 n_mels: int = 40,
                 resnet_aggregation_type: str = 'SAP',
                 resnet_type: str = 'thin', resnet_n_out: int = 512,
                 plot_roc: bool = False,
                 # DataModule args passed in for save_hyperparameters
                 max_sample_length: int = 0,
                 batch_size: int = 128,
                 num_train: int = 0, num_speakers: int = 0,
                 num_workers: int = 1, data_path: str = './data/', rng_seed: int = 0,
                 **kwargs):
        super().__init__()
        # Training/testing params
        self.learning_rate = learning_rate
        self.max_epochs: Final = max_epochs  # Needed for OneCycleLR
        self.augment: Final = augment
        self.specaugment: Final = specaugment

        self.save_hyperparameters('model', 'learning_rate', 'max_epochs',
                                  'batch_size', 'rng_seed', 'max_sample_length',
                                  'num_speakers', 'num_train', 'augment',
                                  'specaugment', 'n_mels',
                                  'resnet_aggregation_type', 'resnet_type', 'resnet_n_out')

        self._plot_roc: Final = plot_roc

        if model == 'snn-angularproto':
            self._example_input_array = torch.rand(4, 1, n_mels, 201)
        else:
            self._example_input_array = [torch.rand(4, 1, n_mels, 201), torch.rand(4, 1, n_mels, 201)]

        self.instancenorm: Final[nn.Module] = nn.InstanceNorm1d(n_mels)
        self.spectrogram: Final[nn.Module] = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=n_fft, win_length=400, hop_length=160,
                                                 window_fn=torch.hamming_window, n_mels=n_mels)
        )

        # Partial SpecAugment, if toggled.
        self.augment_spectrogram: Optional[nn.Module] = None
        if self.specaugment:
            F = 0.20
            T = 0.10
            self.augment_spectrogram = torch.nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=int(F * n_mels)),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=int(F * n_mels)),
                torchaudio.transforms.TimeMasking(time_mask_param=int(T * (n_fft // 2 + 1))),
                torchaudio.transforms.TimeMasking(time_mask_param=int(T * (n_fft // 2 + 1)))
            )

        # waveform -> MelSpectrogram (n_fft=512, n_mels) -> C
        self.cnn: nn.Module
        if resnet_type == 'thin':
            self.cnn = ThinResNet(nOut=resnet_n_out, encoder_type=resnet_aggregation_type, n_mels=n_mels)
        elif resnet_type == 'fast':
            self.cnn = FastResNet(nOut=resnet_n_out, encoder_type=resnet_aggregation_type)
        else:
            raise ValueError

        if resnet_aggregation_type == 'SAP':
            cnn_out_dim = 512
        elif resnet_aggregation_type == 'ASP':
            cnn_out_dim = 512
        elif resnet_aggregation_type.endswith('VLAD'):
            cnn_out_dim = 512

        self.out: nn.Module = nn.Linear(cnn_out_dim, 1)

        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy(compute_on_step=False)
        self.test_accuracy = pl.metrics.Accuracy(compute_on_step=False)

    @staticmethod
    def add_model_specific_args(parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)

        general = parser.add_argument_group('General')
        general.add_argument('--plot_roc', action='store_true', default=False,
                             help='Plot ROC curve after testing.')

        training = parser.add_argument_group('Training/testing')
        training.add_argument('--learning_rate', type=float, default=1e-3,
                              help='Initial learning rate used by auto_lr_find')
        training.add_argument('--specaugment', action='store_true', default=False,
                              help='Augment training data using SpecAugment without time warping.')

        model = parser.add_argument_group('Model')
        model.add_argument('--n_mels', type=int, default=40, help='# of mels to use in the MelSpectrograms.')
        model.add_argument('--n_fft', type=int, default=512, help='size of FFT used in the MelSpectrograms.')

        model.add_argument('--resnet_type', type=str.lower, default='thin',
                           help='Which ResNet to use: thin, fast.')
        model.add_argument('--resnet_n_out', type=int, default=512)
        model.add_argument('--resnet_aggregation_type', type=str, default='SAP',
                           choices=['SAP', 'ASP', 'NetVLAD', 'GhostVLAD'],
                           help='The aggregation method used in ResNet.')

        return parser

    def spectogram_transform(self, x: torch.Tensor, augment: bool = False) -> torch.Tensor:
        with torch.no_grad():
            x = self.spectrogram(x)+1e-6
            x = x.log()
            if augment and self.augment_spectrogram:
                x = self.augment_spectrogram(x)
            x = self.instancenorm(x).unsqueeze(1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate,
                                                        epochs=self.max_epochs,
                                                        steps_per_epoch=len(self.train_dataloader()))
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'step'}]

    def validation_epoch_end(self, val_step_outputs: List[List[torch.Tensor]]):
        self.log('val_acc_epoch', self.val_accuracy.compute())

        try:
            eer, eer_thresh, auc, mdcf, mdcf_thresh = compute_evaluation_metrics(val_step_outputs)
            self.log('val_eer', eer)
            self.log('val_eer_threshold', eer_thresh)
            self.log('val_min_dcf', mdcf)
            self.log('val_min_dcf_threshold', mdcf_thresh)
            self.log('val_auc', auc)
        except ValueError:
            # Will fail if labels are all the same value, which tends to happen with auto_lr_finder.
            # So we just ignore these.
            pass

    def test_epoch_end(self, test_step_outputs: List[List[torch.Tensor]]):
        self.log('test_acc_epoch', self.test_accuracy.compute())

        # Avoid ValueError: No negative samples in targets, false positive value should be meaningless
        if self.trainer.fast_dev_run:
            return

        eer, eer_thresh, auc, mdcf, mdcf_thresh = compute_evaluation_metrics(test_step_outputs, plot=self._plot_roc)
        self.log('test_eer', eer)
        self.log('test_eer_threshold', eer_thresh)
        self.log('test_min_dcf', mdcf)
        self.log('test_min_dcf_threshold', mdcf_thresh)
        self.log('test_auc', auc)
