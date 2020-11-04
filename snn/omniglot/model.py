from typing import Tuple, Optional, Union, List, Any
from typing_extensions import Final
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from snn.omniglot.data_loader import Omniglot, TrainSet, TestSet


class CNNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # 1-channel input
            nn.Conv2d(1, 64, kernel_size=10),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=7),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=4),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)


class TwinNet(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-3,
                            help='Initial learning rate used by auto_lr_find')
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=1, help='# of workers used by DataLoader')
        parser.add_argument('--trials', type=int, default=320, help='# of 1-shot trials (validation/test)')
        parser.add_argument('--way', type=int, default=20, help='# of ways in 1-shot trials')
        parser.add_argument('--num_train', type=int, default=50000,
                            help='# of pairs in training set (augmented, random pairs)')
        parser.add_argument('--data_path', type=str, default='./data/')
        return parser

    # NOTE: Defaults here shouldn't really matter much, they're just here to make initializing the model
    # for other purposes easier (such as log_graph)...
    def __init__(self, learning_rate: float = 1e-3,
                 batch_size: int = 128, max_epochs: int = 100,
                 num_workers: int = 1, data_path: str = './data/', rng_seed: int = 0,
                 way: int = 20, trials: int = 320, num_train: int = 90000,
                 **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs  # Needed for OneCycleLR
        self.save_hyperparameters('learning_rate', 'batch_size', 'max_epochs', 'rng_seed', 'way', 'trials', 'num_train')

        self._way: Final = way
        self._trials: Final = trials
        self._num_train: Final = num_train

        self._data_path: Final = data_path
        self._rng_seed: Final = rng_seed

        # TODO: Does this work right, what about TPUs?
        self._num_workers = num_workers
        self._pin_memory = False
        if torch.cuda.is_available():
            self._num_workers = num_workers
            self._pin_memory = True

        # Conv2d and Linear use kaiming_uniform_ initialization (AKA He-at-al)
        # Source: https://github.com/pytorch/pytorch/tree/master/torch/nn/modules
        self.cnn: nn.Module = CNNLayer()
        # 256*6*6 = 9216
        # NOTE: Amount of parameters seems really high...?
        self.fcl: nn.Module = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.out: nn.Module = nn.Linear(4096, 1)

        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy(compute_on_step=False)
        self.test_accuracy = pl.metrics.Accuracy(compute_on_step=False)

    # Prediction/inference.
    # Basically as is from https://github.com/kevinzakka/one-shot-siamese
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor: # type: ignore[override]
        # print(x1.shape)
        x1 = self.cnn(x1)
        # print('{0}::{1}'.format(x1.shape, x1.size()))
        x1 = x1.view(x1.size()[0], -1)
        # print(x1.shape)
        x1 = self.fcl(x1)

        x2 = self.cnn(x2)
        x2 = x2.view(x2.size()[0], -1)
        x2 = self.fcl(x2)

        # Calculate L1 distance (Manhattan distance) of the twin CNN output vectors.
        dist = torch.abs(x1 - x2)

        # Use a Linear layer to learn the weights for the final distance value (hence "weighted L1 distance"),
        # but omit sigmoid. Because the loss (binary_cross_entropy_with_logits) function handles it for us.
        return self.out(dist)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate,
                                                        epochs=self.max_epochs,
                                                        steps_per_epoch=len(self.train_dataloader()))
        return [optimizer], [ { 'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'step' } ]

    @staticmethod
    def loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(x, y)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # type: ignore[override]
                      batch_idx: int) -> torch.Tensor:
        x1, x2, y = batch  # Train DataLoader output
        out = self.forward(x1, x2)
        loss = self.loss(out, y)

        acc = self.train_accuracy(out, y)
        self.log('train_acc_step', acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_epoch_end(self, training_step_outputs: List[Any]):
        self.log('learning_rate_epoch', self.learning_rate)
        self.log('train_acc_epoch', self.train_accuracy.compute())

        # Graph the model, requires input data for forward(),
        # so we need to do it here as we need a dataloader().
        # FIXME: Does not seem to be doing anything?
        if self.current_epoch == 1:
            x1, x2, _ = next(iter(self.train_dataloader()))
            self.logger.experiment.add_graph(TwinNet(), [x1, x2])

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # type: ignore[override]
                        batch_idx: int):
        x1, x2, y = batch  # Validation DataLoader output
        out = self.forward(x1, x2)
        loss = self.loss(out, y)

        self.val_accuracy(out, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def validation_epoch_end(self, val_step_outputs: List[Any]):
        self.log('val_acc_epoch', self.val_accuracy.compute())

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # type: ignore[override]
                  batch_idx: int):
        x1, x2, y = batch  # Test DataLoader output
        out = self.forward(x1, x2)
        loss = self.loss(out, y)

        self.test_accuracy(out, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

    def test_epoch_end(self, test_step_outputs: List[Any]):
        self.log('test_acc_epoch', self.test_accuracy.compute())

    def prepare_data(self):
        # Download and augment datasets if necessary.
        # NOTE: Should not assing state here.
        Omniglot(data_path=self._data_path, mode='train', download=True)
        Omniglot(data_path=self._data_path, mode='valid', download=True)
        Omniglot(data_path=self._data_path, mode='test', download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            train_dataset = Omniglot(data_path=self._data_path, mode='train', download=False)
            val_dataset = Omniglot(data_path=self._data_path, mode='valid', download=False)
            self.training_set = TrainSet(train_dataset, augment=True,
                                         num_train=self._num_train)
            self.validation_set = TestSet(val_dataset, seed=self._rng_seed,
                                          trials=self._trials, way=self._way)
        if stage == 'test' or stage is None:
            test_dataset = Omniglot(data_path=self._data_path, mode='test', download=False)
            self.test_set = TestSet(test_dataset, seed=self._rng_seed,
                                    trials=self._trials, way=self._way)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.training_set, batch_size=self.batch_size,
            shuffle=True, num_workers=self._num_workers, pin_memory=self._pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.validation_set, batch_size=self._way, shuffle=False,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_set, batch_size=self._way, shuffle=False,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
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
        filepath='./snn-omniglot-{epoch}-{val_loss:.2f}',
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
