import argparse
from typing import cast, Tuple, Optional, Union, List, Any
from typing_extensions import Final
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.datasets as dset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.metrics.functional import roc
from pytorch_lightning.metrics.functional.classification import auroc
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt

from snn.librispeech.data_loader import PairDataset, TripletDataset
from resnet.ResNetSE34V2 import MainModel as ResNet
from resnet.utils import PreEmphasis
from capsnet.CapsNet import CapsNetWithoutPrimaryCaps, MarginLoss


def minDCF(fpr, fnr, thresholds, p_target: float = 0.05, c_miss: int = 1, c_fa: int = 1):
    c_det = (c_miss * p_target * fnr) + (c_fa * (1 - p_target) * fpr)
    min_c_det_idx = torch.argmin(c_det)
    min_c_det = c_det[min_c_det_idx]
    min_c_det_threshold = thresholds[min_c_det_idx]

    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def

    return min_dcf, min_c_det_threshold


def equal_error_rate(fpr: torch.Tensor, fnr: torch.Tensor,
                     thresholds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Index of the nearest intersection point.
    idx = torch.argmin(torch.abs(fnr - fpr))

    # roc() can return fpr, tpr, thresholds with much shorter length than scores.size(0)
    # (e.g. cases where scores.size(0) = 2611 and fnr.size(0) = 4), which leads to a large
    # differences between fpr[e] and fnr[e]. Thus we take the mean of fpr[e] and fnr[e] as a
    # good enough approximation. Could also interpolate using scipy brentq and interp1d methods,
    # but this on the other hand leads to frequent division by zero errors in some cases.
    eer = 0.5 * (fpr[idx] + fnr[idx])
    eer_threshold = thresholds[idx]

    # https://yangcha.github.io/EER-ROC/
    # https://stackoverflow.com/questions/28339746/equal-error-rate-in-python
    # Unfortunately this seems to fail frequently, incorrectly returning 0.0 and nan.
    #
    # from scipy.optimize import brentq
    # from scipy.interpolate import interp1d
    # eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # eer_threshold = interp1d(fpr, thresholds)(eer)

    return eer, eer_threshold, idx


def compute_evaluation_metrics(outputs: List[List[torch.Tensor]],
                               plot: bool = False) -> Tuple[torch.Tensor, torch.Tensor,
                                                            torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.cat(list((scores for step in outputs for scores in step[0])))
    # NOTE: Need sigmoid here because we skip the sigmoid in forward() due to using BCE with logits for loss.
    scores = torch.sigmoid(scores)
    labels = torch.cat(list((labels for step in outputs for labels in step[1])))
    auc = auroc(scores, labels, pos_label=1)
    fpr, tpr, thresholds = roc(scores, labels, pos_label=1)
    fnr = 1 - tpr

    eer, eer_threshold, idx = equal_error_rate(fpr, fnr, thresholds)
    min_dcf, min_dcf_threshold = minDCF(fpr, fnr, thresholds)

    if plot:
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.plot([0, 1], [0, 1], 'r--', label='Reference', alpha=0.6)
        plt.plot([1, 0], [0, 1], 'k--', label='EER line', alpha=0.6)
        plt.plot(fpr, tpr, label='ROC curve')
        plt.fill_between(fpr, tpr, 0, label='AUC', color='0.8')
        plt.plot(fpr[idx], tpr[idx], 'ko', label='EER = {:.2f}%'.format(eer * 100))  # EER point
        plt.legend()
        plt.show()

    return eer, eer_threshold, auc, min_dcf, min_dcf_threshold


def collate_var_len_tuples_fn(batch):
    a, b, labels = zip(*batch)
    lengths = torch.as_tensor(list(map(lambda t1, t2: (t1.size(0), t2.size(0)), a, b)))
    a = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
    b = torch.nn.utils.rnn.pad_sequence(b, batch_first=True)
    return a, b, lengths, torch.utils.data.dataloader.default_collate(labels)


class TwinNet(pl.LightningModule):
    # NOTE: Defaults here shouldn't really matter much, they're just here to make initializing the model
    # for other purposes easier (such as log_graph)...
    def __init__(self, learning_rate: float = 1e-3,
                 max_sample_length: int = 0,
                 batch_size: int = 128, max_epochs: int = 100,
                 num_train: int = 0, num_speakers: int = 0,
                 num_workers: int = 1, data_path: str = './data/', rng_seed: int = 0,
                 n_mels: int = 40, aggregation_type: str = 'SAP',
                 augment: bool = False, plot_roc: bool = False,
                 **kwargs):
        super().__init__()
        # Training/testing params
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs: Final = max_epochs  # Needed for OneCycleLR
        self.max_sample_length: Final = None if max_sample_length == 0 else max_sample_length
        self.rng_seed: Final = rng_seed
        self.augment: Final = augment
        self.num_train: Final = num_train
        self.num_speakers: Final = None if num_speakers == 0 else num_speakers

        self.save_hyperparameters('learning_rate', 'batch_size', 'max_epochs', 'rng_seed', 'max_sample_length',
                                  'num_speakers', 'num_train', 'augment', 'n_mels', 'aggregation_type')

        self._plot_roc: Final = plot_roc
        self._data_path: Final = data_path

        # Pad samples in DataLoader batches to the same length as the longest sample.
        self._collate_fn: Final = None if self.max_sample_length else collate_var_len_tuples_fn

        # TODO: Does this work right, what about TPUs?
        self._num_workers = num_workers
        self._pin_memory = False
        if torch.cuda.is_available():
            self._num_workers = num_workers
            self._pin_memory = True

        n_fft = 512
        self.instancenorm = nn.InstanceNorm1d(n_mels)
        self.spectrogram: nn.Module = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=n_fft, win_length=400, hop_length=160,
                                                 window_fn=torch.hamming_window, n_mels=n_mels)
        )

        self.augment_spectrogram: Optional[nn.Module] = None
        if self.augment:
            F = 0.25
            T = 0.15
            self.augment_spectrogram = torch.nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=int(F * n_mels)),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=int(F * n_mels)),
                torchaudio.transforms.TimeMasking(time_mask_param=int(T * (n_fft // 2 + 1))),
                torchaudio.transforms.TimeMasking(time_mask_param=int(T * (n_fft // 2 + 1)))
            )

        # waveform -> MelSpectrogram (n_fft=512, n_mels) -> C
        self.cnn: nn.Module = ResNet(nOut=512, encoder_type=aggregation_type, n_mels=n_mels)

        if aggregation_type == 'SAP':
            cnn_out_dim = 512
        elif aggregation_type == 'ASP':
            cnn_out_dim = 1024
        elif aggregation_type.endswith('VLAD'):
            cnn_out_dim = 512

        # 2xC -> 4x128
        self.caps: nn.Module = CapsNetWithoutPrimaryCaps(routing_iterations=3,
                                                         input_caps=2, input_dim=cnn_out_dim,
                                                         output_caps=4, output_dim=128)

        # C -> 1, or if CapsNet (4*128) -> 1
        self.out: nn.Module = nn.Linear(cnn_out_dim, 1)

        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy(compute_on_step=False)
        self.test_accuracy = pl.metrics.Accuracy(compute_on_step=False)

        self.training_set: Dataset
        self.validation_set: Dataset
        self.test_set: Dataset
        self.training_sampler: Optional[Sampler] = None

    @staticmethod
    def add_model_specific_args(parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)

        general = parser.add_argument_group('General')
        general.add_argument('--num_workers', type=int, default=1, help='# of workers used by DataLoader.')
        general.add_argument('--data_path', type=str, default='./data/')
        general.add_argument('--plot_roc', action='store_true', default=False,
                             help='Plot ROC curve after testing.')
        general.add_argument('--rng_seed', type=int, default=1)

        training = parser.add_argument_group('Training/testing')
        training.add_argument('--learning_rate', type=float, default=1e-3,
                              help='Initial learning rate used by auto_lr_find')
        training.add_argument('--batch_size', type=int, default=128)
        training.add_argument('--num_speakers', type=int, default=0,
                              help='Limits the # of speakers to train on, 0 to select all.')
        training.add_argument('--num_train', type=int, default=0,
                              help='# of samples to take from training data each epoch, 0 to use all.'
                              'Use with --augment if value is greater than the amount of training data pairs.')
        training.add_argument('--augment', action='store_true', default=False,
                              help='Augment training data using SpecAugment without time warping.')
        training.add_argument('--max_sample_length', type=int, default=2,
                              help='Maximum length in seconds of samples used, clipped/padded to fit. 0 for no limit.')

        model = parser.add_argument_group('Model')
        model.add_argument('--n_mels', type=int, default=40)
        model.add_argument('--aggregation_type', type=str, default='SAP',
                           help='The aggregation method used in ResNet. Available types: SAP, ASP, NetVLAD, GhostVLAD.')
        return parser

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # print(x1.shape)
        x1 = self.cnn(x1)
        # print("{0}::{1}".format(x1.shape, x1.size()))
        x2 = self.cnn(x2)

        # x1 = F.normalize(x1, p=2, dim=1)
        # x2 = F.normalize(x2, p=2, dim=1)
        # x = torch.stack((x1, x2), dim=1)
        # x, _ = self.caps(x)
        # x = torch.flatten(x, 1)
        # # print(x.shape)
        # return self.out(x)

        # Using a capsule network instead of a plain distance function...
        dist = torch.abs(x1 - x2)
        #dist = torch.abs(x[0] - x[1])
        #dist = torch.abs(x[:, 0] - x[:, 1])
        return self.out(dist)

    def spectogram_transform(self, x: torch.Tensor, augment: bool = False) -> torch.Tensor:
        with torch.no_grad():
            x = self.spectrogram(x)+1e-6
            x = x.log()
            if augment and self.augment_spectrogram:
                x = self.augment_spectrogram(x)
            x = self.instancenorm(x).unsqueeze(1)
        return x

    def configure_optimizers(self):
        assert self.num_train == 0 or len(self.train_dataloader()) == self.num_train
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate,
                                                        epochs=self.max_epochs,
                                                        steps_per_epoch=len(self.train_dataloader()))
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'step'}]

    @staticmethod
    def loss(x: torch.Tensor, y: torch.Tensor):
        return F.binary_cross_entropy_with_logits(x, y)

    def training_step(self,  # type: ignore[override]
                      batch: Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                   Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                      batch_idx: int) -> torch.Tensor:
        if self.max_sample_length is None:
            x1, x2, lengths, y = cast(Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch)
            # TODO: Do something to ignore padding?
        else:
            x1, x2, y = cast(Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch)

        x1 = self.spectogram_transform(x1, augment=self.augment)
        x2 = self.spectogram_transform(x2, augment=self.augment)
        out = self(x1, x2)
        loss = self.loss(out, y)

        acc = self.train_accuracy(out, y)
        self.log('train_acc_step', acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def training_epoch_end(self, training_step_outputs: List[Any]):
        self.log('train_acc_epoch', self.train_accuracy.compute())

    def validation_step(self,  # type: ignore[override]
                        batch: Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                     Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                        batch_idx: int):
        if self.max_sample_length is None:
            x1, x2, lengths, y = cast(Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch)
            # TODO: Do something to ignore padding?
        else:
            x1, x2, y = cast(Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch)

        x1 = self.spectogram_transform(x1)
        x2 = self.spectogram_transform(x2)
        out = self(x1, x2)
        loss = self.loss(out, y)

        self.val_accuracy(out, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return [out, y]

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

    def test_step(self,  # type: ignore[override]
                  batch: Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                               Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                  batch_idx: int):
        if self.max_sample_length is None:
            x1, x2, lengths, y = cast(Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch)
            # TODO: Do something to ignore padding?
        else:
            x1, x2, y = cast(Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch)

        x1 = self.spectogram_transform(x1)
        x2 = self.spectogram_transform(x2)
        out = self(x1, x2)
        loss = self.loss(out, y)

        self.test_accuracy(out, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

        return [out, y]

    def test_epoch_end(self, test_step_outputs: List[List[torch.Tensor]]):
        self.log('test_acc_epoch', self.test_accuracy.compute())

        eer, eer_thresh, auc, mdcf, mdcf_thresh = compute_evaluation_metrics(test_step_outputs, plot=self._plot_roc)
        self.log('test_eer', eer)
        self.log('test_eer_threshold', eer_thresh)
        self.log('test_min_dcf', mdcf)
        self.log('test_min_dcf_threshold', mdcf_thresh)
        self.log('test_auc', auc)

    def prepare_data(self):
        dset.LIBRISPEECH(self._data_path, url='train-clean-100', download=True)
        dset.LIBRISPEECH(self._data_path, url='dev-clean', download=True)
        dset.LIBRISPEECH(self._data_path, url='test-clean', download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            train_dataset = dset.LIBRISPEECH(self._data_path, url='train-clean-100', download=False)
            val_dataset = dset.LIBRISPEECH(self._data_path, url='dev-clean', download=False)

            self.training_set = PairDataset(train_dataset, n_speakers=self.num_speakers,
                                            max_sample_length=self.max_sample_length)
            if self.num_train != 0:
                self.training_sampler = torch.utils.data.RandomSampler(self.training_set, replacement=True,
                                                                       num_samples=self.num_train)

            self.validation_set = PairDataset(val_dataset, max_sample_length=self.max_sample_length)
        if stage == 'test' or stage is None:
            test_dataset = dset.LIBRISPEECH(self._data_path, url='test-clean', download=False)
            self.test_set = PairDataset(test_dataset, max_sample_length=self.max_sample_length)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.training_set, batch_size=self.batch_size,
            shuffle=self.training_sampler is None,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            sampler=self.training_sampler,
            collate_fn=self._collate_fn,  # type: ignore
            worker_init_fn=lambda worker_id: pl.seed_everything(worker_id + self.rng_seed)  # type: ignore
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.validation_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            collate_fn=self._collate_fn,  # type: ignore
            worker_init_fn=lambda worker_id: pl.seed_everything(worker_id + self.rng_seed)  # type: ignore
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            collate_fn=self._collate_fn,  # type: ignore
            worker_init_fn=lambda worker_id: pl.seed_everything(worker_id + self.rng_seed)  # type: ignore
        )


def train_and_test(args: argparse.Namespace):
    dict_args = vars(args)
    seed = args.rng_seed
    log_dir = args.log_dir
    early_stop = args.early_stop
    early_stop_min_delta = args.early_stop_min_delta
    early_stop_patience = args.early_stop_patience

    pl.seed_everything(seed)
    callbacks: List[pl.callbacks.Callback] = [
        LearningRateMonitor(logging_interval='step')
    ]

    if early_stop:
        # Should give enough time for lr_scheduler to try do it's thing.
        callbacks.append(EarlyStopping(
            monitor='val_loss', mode='min',
            min_delta=early_stop_min_delta, patience=early_stop_patience,
            verbose=True, strict=True
        ))

    checkpoint_callback = ModelCheckpoint(
        # TODO: Is low val_loss the best choice for choosing the best model?
        monitor='val_loss', mode='min',
        filepath='./checkpoints/snn-librispeech-{epoch}-{val_loss:.2f}',
        save_top_k=3
    )

    logger = TensorBoardLogger(log_dir, name='snn')
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, progress_bar_refresh_rate=20,
                                            deterministic=True, auto_lr_find=True,
                                            checkpoint_callback=checkpoint_callback,
                                            callbacks=callbacks)
    model = TwinNet(**dict_args)

    # Tune learning rate.
    trainer.tune(model)
    logger.log_hyperparams(params=model.hparams)

    # Train model.
    trainer.fit(model)
    print('Best model saved to: ', checkpoint_callback.best_model_path)

    # Test using best checkpoint.
    trainer.test()
