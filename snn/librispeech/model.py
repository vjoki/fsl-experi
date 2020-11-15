from typing import cast, Tuple, Optional, Union, List, Any
from typing_extensions import Final
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torchaudio.datasets as dset
from torch.utils.data import DataLoader

from snn.librispeech.data_loader import PairDataset, TripletDataset
from resnet.ResNetSE34V2 import MainModel as ResNet
from capsnet.CapsNet import CapsNetWithoutPrimaryCaps, MarginLoss


def collate_var_len_tuples_fn(batch):
    a, b, labels = zip(*batch)
    lengths = torch.tensor([(t1.size(0), t2.size(0)) for (t1, t2) in zip(a, b)])
    a = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
    b = torch.nn.utils.rnn.pad_sequence(b, batch_first=True)
    return a, b, lengths, torch.utils.data.dataloader.default_collate(labels)


class TwinNet(pl.LightningModule):
    # NOTE: Defaults here shouldn't really matter much, they're just here to make initializing the model
    # for other purposes easier (such as log_graph)...
    def __init__(self, learning_rate: float = 1e-3,
                 max_sample_length: Optional[int] = None,
                 batch_size: int = 128, max_epochs: int = 100,
                 num_workers: int = 1, data_path: str = './data/', rng_seed: int = 0,
                 **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs  # Needed for OneCycleLR
        self.save_hyperparameters('learning_rate', 'batch_size', 'max_epochs', 'rng_seed', 'max_sample_length')

        self._max_sample_length: Final = max_sample_length
        self._data_path: Final = data_path
        self._rng_seed: Final = rng_seed

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

        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy(compute_on_step=False)
        self.test_accuracy = pl.metrics.Accuracy(compute_on_step=False)

    @staticmethod
    def add_model_specific_args(parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-3,
                            help='Initial learning rate used by auto_lr_find')
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=1, help='# of workers used by DataLoader')
        parser.add_argument('--data_path', type=str, default='./data/')
        parser.add_argument('--max_sample_length', type=int, default=None,
                            help='Maximum length of audio samples used, longer samples are clipped to fit.')
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
        if self._max_sample_length is None:
            x1, x2, lengths, y = cast(Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch)
            # TODO: Do something to ignore padding?
        else:
            x1, x2, y = cast(Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch)

        out = self.forward(x1, x2)
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
        if self._max_sample_length is None:
            x1, x2, lengths, y = cast(Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch)
            # TODO: Do something to ignore padding?
        else:
            x1, x2, y = cast(Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch)

        out = self.forward(x1, x2)
        loss = self.loss(out, y)

        self.val_accuracy(out, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def validation_epoch_end(self, val_step_outputs: List[Any]):
        self.log('val_acc_epoch', self.val_accuracy.compute())

    def test_step(self,  # type: ignore[override]
                  batch: Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                               Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                  batch_idx: int):
        if self._max_sample_length is None:
            x1, x2, lengths, y = cast(Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch)
            # TODO: Do something to ignore padding?
        else:
            x1, x2, y = cast(Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch)

        out = self.forward(x1, x2)
        loss = self.loss(out, y)

        self.test_accuracy(out, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

    def test_epoch_end(self, test_step_outputs: List[Any]):
        self.log('test_acc_epoch', self.test_accuracy.compute())

    def prepare_data(self):
        dset.LIBRISPEECH(self._data_path, url='train-clean-100', download=True)
        dset.LIBRISPEECH(self._data_path, url='dev-clean', download=True)
        dset.LIBRISPEECH(self._data_path, url='test-clean', download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            train_dataset = dset.LIBRISPEECH(self._data_path, url='train-clean-100', download=False)
            val_dataset = dset.LIBRISPEECH(self._data_path, url='dev-clean', download=False)
            self.training_set = PairDataset(train_dataset, max_sample_length=self._max_sample_length)
            self.validation_set = PairDataset(val_dataset, max_sample_length=self._max_sample_length)
        if stage == 'test' or stage is None:
            test_dataset = dset.LIBRISPEECH(self._data_path, url='test-clean', download=False)
            self.test_set = PairDataset(test_dataset, max_sample_length=self._max_sample_length)

    def train_dataloader(self) -> DataLoader:
        collate_fn = None
        if self._max_sample_length is None:
            collate_fn = collate_var_len_tuples_fn

        return DataLoader(
            self.training_set, batch_size=self.batch_size,
            shuffle=True, num_workers=self._num_workers, pin_memory=self._pin_memory,
            collate_fn=collate_fn
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        collate_fn = None
        if self._max_sample_length is None:
            collate_fn = collate_var_len_tuples_fn

        return DataLoader(
            self.validation_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            collate_fn=collate_fn
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        collate_fn = None
        if self._max_sample_length is None:
            collate_fn = collate_var_len_tuples_fn

        return DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            collate_fn=collate_fn
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
