import argparse
from typing import cast, Callable, Tuple, Optional, Union, List, Any
from typing_extensions import Final
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.metrics.functional.classification import roc, auroc
import torchaudio.datasets as dset
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataset import Dataset

from snn.librispeech.data_loader import PairDataset, TripletDataset
from resnet.ResNetSE34V2 import MainModel as ResNet
from resnet.utils import PreEmphasis
from capsnet.CapsNet import CapsNetWithoutPrimaryCaps, MarginLoss


def equal_error_rate(scores: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    fpr, tpr, thresholds = roc(scores, labels, pos_label=1)
    fnr = 1 - tpr

    x = torch.argmin(torch.abs(fnr - fpr))
    eer_threshold = thresholds[x]
    eer = fpr[x]

    return eer, eer_threshold


def compute_epoch_end_eer(outputs: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.cat(list((scores for step in outputs for scores in step[0])))
    scores = torch.sigmoid(scores)
    labels = torch.cat(list((labels for step in outputs for labels in step[1])))
    auc = auroc(scores, labels, pos_label=1)
    eer, threshold = equal_error_rate(scores, labels)
    return eer, threshold, auc


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
                 max_sample_length: Optional[int] = None,
                 batch_size: int = 128, max_epochs: int = 100, num_train: int = 0,
                 num_workers: int = 1, data_path: str = './data/', rng_seed: int = 0,
                 augment: bool = False,
                 **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs: Final = max_epochs  # Needed for OneCycleLR
        self.max_sample_length: Final = None if max_sample_length == 0 else max_sample_length
        self.rng_seed: Final = rng_seed
        self.augment: Final = augment
        self.num_train: Final = num_train
        self.save_hyperparameters('learning_rate', 'batch_size', 'max_epochs', 'rng_seed', 'max_sample_length',
                                  'num_train', 'augment')

        self._data_path: Final = data_path

        # Pad samples in DataLoader batches to the same length as the longest sample.
        self._collate_fn: Final = None if self.max_sample_length else collate_var_len_tuples_fn

        # TODO: Does this work right, what about TPUs?
        self._num_workers = num_workers
        self._pin_memory = False
        if torch.cuda.is_available():
            self._num_workers = num_workers
            self._pin_memory = True

        # waveform -> MelSpectrogram (n_fft=512, n_mels=40) -> 512
        self.cnn: nn.Module = ResNet(nOut=512, encoder_type='SAP')
        # 2x512 -> 4x128
        self.caps: nn.Module = CapsNetWithoutPrimaryCaps(routing_iterations=3,
                                                         input_caps=2, input_dim=512,
                                                         output_caps=4, output_dim=128)
        # 4x128 -> 512 -> 1
        self.out: nn.Module = nn.Linear(512, 1)

        n_mels = 40
        n_fft = 512
        self.instancenorm = nn.InstanceNorm1d(n_mels)
        self.spectrogram = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=n_fft, win_length=400, hop_length=160,
                                                 window_fn=torch.hamming_window, n_mels=n_mels)
        )

        self.augment_spectogram: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        if self.augment:
            F = 0.25
            T = 0.15
            self.augment_spectogram = torch.nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=int(F * n_mels)),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=int(F * n_mels)),
                torchaudio.transforms.TimeMasking(time_mask_param=int(T * (n_fft // 2 + 1))),
                torchaudio.transforms.TimeMasking(time_mask_param=int(T * (n_fft // 2 + 1)))
            )

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
        parser.add_argument('--learning_rate', type=float, default=1e-3,
                            help='Initial learning rate used by auto_lr_find')
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_train', type=int, default=0,
                            help='# of samples to take from training data each epoch, 0 to use all.'
                            'Use with --augment if value is greater than the amount of training data pairs.')
        parser.add_argument('--augment', action='store_true', default=False,
                            help='Augment training data using SpecAugment without time warping.')
        parser.add_argument('--num_workers', type=int, default=1, help='# of workers used by DataLoader.')
        parser.add_argument('--data_path', type=str, default='./data/')
        parser.add_argument('--max_sample_length', type=int, default=32000,
                            help='Maximum length of samples used, clipped to fit. Set to 0 for no limit.')
        return parser

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # print(x1.shape)
        x1 = self.cnn(x1)
        # print("{0}::{1}".format(x1.shape, x1.size()))
        x2 = self.cnn(x2)

        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
        x = torch.stack((x1, x2), dim=1)
        x, _ = self.caps(x)
        x = torch.flatten(x, 1)
        # print(x.shape)

        # Using a capsule network instead of a plain distance function...
        #dist = torch.abs(x1 - x2)
        #dist = torch.abs(x[0] - x[1])
        #dist = torch.abs(x[:, 0] - x[:, 1])

        #return self.out(dist)
        return self.out(x)

    def spectogram_transform(self, x: torch.Tensor, augment: bool = False) -> torch.Tensor:
        with torch.no_grad():
            x = self.spectrogram(x)+1e-6
            x = x.log()
            if augment and self.augment_spectogram:
                x = self.augment_spectogram(x)
            x = self.instancenorm(x).unsqueeze(1)
        return x

    def configure_optimizers(self):
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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
            eer, _, auc = compute_epoch_end_eer(val_step_outputs)
            self.log('val_eer', eer, prog_bar=True)
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

        eer, _, auc = compute_epoch_end_eer(test_step_outputs)
        self.log('test_eer', eer)
        self.log('test_auc', auc)

    def prepare_data(self):
        dset.LIBRISPEECH(self._data_path, url='train-clean-100', download=True)
        dset.LIBRISPEECH(self._data_path, url='dev-clean', download=True)
        dset.LIBRISPEECH(self._data_path, url='test-clean', download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            train_dataset = dset.LIBRISPEECH(self._data_path, url='train-clean-100', download=False)
            val_dataset = dset.LIBRISPEECH(self._data_path, url='dev-clean', download=False)

            self.training_set = PairDataset(train_dataset, max_sample_length=self.max_sample_length)
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
