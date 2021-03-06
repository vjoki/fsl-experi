from typing import Optional, List
import argparse
from typing_extensions import Final
import torch
import torch.nn as nn
import pytorch_lightning as pl

# from pytorch_metric_learning.losses import ContrastiveLoss
# from pytorch_metric_learning.distances import LpDistance
from resnet.ResNetSE34V2 import MainModel as ThinResNet
from resnet.ResNetSE34L import MainModel as FastResNet
from snn.librispeech.utils import compute_evaluation_metrics
from .preprocessor import PreProcessor


class BaseNet(pl.LightningModule):
    SAMPLE_RATE: Final[int] = 16000

    # NOTE: Defaults here shouldn't really matter much, they're just here to make initializing the model
    # for other purposes easier (such as log_graph)...
    def __init__(self, model: str,
                 max_epochs: int = 100,
                 torch_augment: bool = False,
                 augment: bool = False,
                 specaugment: bool = False,
                 signal_transform: str = 'melspectrogram',
                 n_fft: int = 512,
                 n_mels: int = 40,
                 resnet_aggregation_type: str = 'SAP',
                 resnet_type: str = 'thin', resnet_n_out: int = 512,
                 plot_roc: bool = False,
                 # DataModule args passed in for save_hyperparameters
                 max_sample_length: int = 0,
                 batch_size: int = 128,
                 train_batch_size: Optional[int] = None,
                 num_ways: int = 1, num_shots: int = 1,
                 num_train: int = 0, num_speakers: int = 0,
                 num_workers: int = 1, data_path: str = './data/', rng_seed: int = 0,
                 **kwargs):
        super().__init__()
        # Training/testing params
        self.max_epochs: Final = max_epochs  # Needed for OneCycleLR
        if augment and torch_augment:
            raise ValueError('Both augment and torch_augment provided, please choose one.')
        self.augment: Final = augment or torch_augment
        self.specaugment: Final = specaugment
        self.batch_size: Final[int] = batch_size
        self.train_batch_size: Final[int] = train_batch_size or batch_size
        self.signal_transform: Final = signal_transform

        self.num_ways: Final[int] = num_ways
        self.num_shots: Final[int] = num_shots

        self.save_hyperparameters('model', 'max_epochs',
                                  'batch_size', 'train_batch_size', 'rng_seed',
                                  'max_sample_length',
                                  'num_ways', 'num_shots',
                                  'num_speakers', 'num_train',
                                  'augment', 'torch_augment',
                                  'specaugment', 'signal_transform', 'n_fft', 'n_mels',
                                  'resnet_aggregation_type', 'resnet_type', 'resnet_n_out')

        self._plot_roc: Final = plot_roc

        if model in ('snn-angularproto', 'snn-softmaxproto'):
            self._example_input_array = torch.rand(batch_size, 1, n_mels, 201)
        else:
            self._example_input_array = [torch.rand(batch_size, 1, n_mels, 201), torch.rand(batch_size, 1, n_mels, 201)]

        self.spectogram_transform: Final[nn.Module] = PreProcessor(signal_transform,
                                                                   sample_rate=self.SAMPLE_RATE,
                                                                   n_fft=n_fft, n_mels=n_mels,
                                                                   specaugment=specaugment, torch_augment=torch_augment,
                                                                   **kwargs)

        # Bx1xN_MELSxTIME -> BxC
        self.cnn: nn.Module
        if resnet_type == 'thin':
            self.cnn = ThinResNet(nOut=resnet_n_out, encoder_type=resnet_aggregation_type, n_mels=n_mels)
        elif resnet_type == 'fast':
            self.cnn = FastResNet(nOut=resnet_n_out, encoder_type=resnet_aggregation_type)
        else:
            raise ValueError

        cnn_out_dim: int
        if resnet_aggregation_type == 'SAP':
            cnn_out_dim = 512
        elif resnet_aggregation_type == 'ASP':
            cnn_out_dim = 512
        elif resnet_aggregation_type.endswith('VLAD'):
            cnn_out_dim = 512
        self.cnn_out_dim = cnn_out_dim

        if model in ('snn', 'snn-capsnet'):
            # BxC -> Bx1
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
        training.add_argument('--specaugment', action='store_true', default=False,
                              help='Augment training data using SpecAugment without time warping.')
        training.add_argument('--torch_augment', action='store_true', default=False,
                              help='Augment training data using GPU accelerated augmentations.')

        model = parser.add_argument_group('Model')
        model.add_argument('--signal_transform', type=str, default='melspectrogram',
                           choices=['melspectrogram', 'spectrogram', 'mfcc'],
                           help='Waveform signal transform function to use.')
        model.add_argument('--n_mels', type=int, default=40, help='# of mels to use in the MelSpectrograms.')
        model.add_argument('--n_fft', type=int, default=512, help='size of FFT used in the MelSpectrograms.')

        model.add_argument('--resnet_type', type=str.lower, default='thin',
                           help='Which ResNet to use: thin, fast.')
        model.add_argument('--resnet_n_out', type=int, default=512)
        model.add_argument('--resnet_aggregation_type', type=str, default='SAP',
                           choices=['SAP', 'ASP', 'NetVLAD', 'GhostVLAD'],
                           help='The aggregation method used in ResNet.')

        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.get('learning_rate', 1e-3),
                                      weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.get('max_learning_rate', 0.1),
                                                        epochs=self.max_epochs,
                                                        steps_per_epoch=len(self.train_dataloader()))
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'step'}]

    def validation_epoch_end(self, val_step_outputs: List[List[torch.Tensor]]):
        self.log('val_acc_epoch', self.val_accuracy.compute())

        try:
            metrics = compute_evaluation_metrics(val_step_outputs, prefix='val')
            self.log_dict(metrics)
        except ValueError:
            # Will fail if labels are all the same value, which tends to happen with auto_lr_finder.
            # So we just ignore these.
            pass

    def test_epoch_end(self, test_step_outputs: List[List[torch.Tensor]]):
        self.log('test_acc_epoch', self.test_accuracy.compute())

        # Avoid ValueError: No negative samples in targets, false positive value should be meaningless
        if self.trainer.fast_dev_run:
            return

        metrics = compute_evaluation_metrics(test_step_outputs, plot=self._plot_roc, prefix='test')
        self.log_dict(metrics)
