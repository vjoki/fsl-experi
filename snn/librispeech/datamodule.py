import argparse
from typing import Optional
from typing_extensions import Final
import torch
import torchaudio.datasets as dset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataset import Dataset

from snn.librispeech.dataset import PairDataset, NShotKWayDataset
from snn.librispeech.utils import collate_var_len_tuples_fn
from pytorch_lightning.utilities import move_data_to_device


class LibriSpeechDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_set_type: str = 'pair',
                 max_sample_length: int = 0,
                 batch_size: int = 128,
                 train_batch_size: Optional[int] = None,
                 num_ways: int = 1, num_shots: int = 1,
                 num_train: int = 0, num_speakers: int = 0,
                 num_workers: int = 1, data_path: str = './data/', rng_seed: int = 0,
                 augment: bool = False,
                 **kwargs):
        super().__init__()
        self.train_set_type: Final = train_set_type
        self.batch_size = batch_size
        self.train_batch_size: Final[int] = train_batch_size or batch_size

        self.max_sample_length: Final = None if max_sample_length == 0 else max_sample_length
        self.rng_seed: Final = rng_seed
        self.augment: Final = augment
        self.num_train: Final = num_train
        self.num_speakers: Final = None if num_speakers == 0 else num_speakers
        self.num_ways: Final[int] = num_ways
        self.num_shots: Final[int] = num_shots

        self._data_path: Final = data_path

        # Pad samples in DataLoader batches to the same length as the longest sample.
        self._collate_fn: Final = None if self.max_sample_length else collate_var_len_tuples_fn

        # TODO: Does this work right, what about TPUs?
        self._num_workers = num_workers
        self._pin_memory = False
        if torch.cuda.is_available():
            self._num_workers = num_workers
            self._pin_memory = True

        self.training_set: Dataset
        self.validation_set: Dataset
        self.test_set: Dataset
        self.training_sampler: Optional[Sampler] = None

    # https://github.com/PyTorchLightning/pytorch-lightning/issues/4270
    def transfer_batch_to_device(self, batch, device):
        return move_data_to_device(batch, device)

    @staticmethod
    def add_dataset_specific_args(parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)

        general = parser.add_argument_group('General')
        general.add_argument('--num_workers', type=int, default=1, help='# of workers used by DataLoader.')
        general.add_argument('--data_path', type=str, default='./data/')
        general.add_argument('--rng_seed', type=int, default=1)

        training = parser.add_argument_group('Training/testing')
        training.add_argument('--batch_size', type=int, default=128)
        training.add_argument('--max_sample_length', type=int, default=2,
                              help='Maximum length in seconds of samples used, clipped/padded to fit. 0 for no limit.')

        training = parser.add_argument_group('Training')
        training.add_argument('--train_batch_size', type=int, default=None,
                              help='Optionally use different batch size for training.')
        training.add_argument('--num_ways', type=int, default=5,
                              help='# of ways to train with (nshotkway only).')
        training.add_argument('--num_shots', type=int, default=2,
                              help='# of shots for each way (nshotkway only).')
        training.add_argument('--num_speakers', type=int, default=0,
                              help='Limits the # of speakers to train on, 0 to select all.')
        training.add_argument('--num_train', type=int, default=0,
                              help='# of samples to take from training data each epoch, 0 to use all.'
                              'Use with --augment if value is greater than the amount of training data pairs.')
        training.add_argument('--augment', action='store_true', default=False,
                              help='Augment training data by adding noise (gaussian or RIR).')

        return parser

    def prepare_data(self):
        dset.LIBRISPEECH(self._data_path, url='train-clean-100', download=True)
        dset.LIBRISPEECH(self._data_path, url='dev-clean', download=True)
        dset.LIBRISPEECH(self._data_path, url='test-clean', download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            train_dataset = dset.LIBRISPEECH(self._data_path, url='train-clean-100', download=False)
            val_dataset = dset.LIBRISPEECH(self._data_path, url='dev-clean', download=False)

            if self.train_set_type == 'pair':
                self.training_set = PairDataset(train_dataset, n_speakers=self.num_speakers, augment=self.augment,
                                                max_sample_length=self.max_sample_length)
            elif self.train_set_type == 'nshotkway':
                self.training_set = NShotKWayDataset(train_dataset,
                                                     num_shots=self.num_shots, num_ways=self.num_ways,
                                                     augment=self.augment,
                                                     max_sample_length=self.max_sample_length)

            if self.num_train != 0:
                self.training_sampler = torch.utils.data.RandomSampler(self.training_set, replacement=True,
                                                                       num_samples=self.num_train)

            self.validation_set = PairDataset(val_dataset, max_sample_length=self.max_sample_length)
        if stage == 'test' or stage is None:
            test_dataset = dset.LIBRISPEECH(self._data_path, url='test-clean', download=False)
            self.test_set = PairDataset(test_dataset, max_sample_length=self.max_sample_length)

    def worker_init(self, worker_id):
        pl.seed_everything(worker_id + self.rng_seed)

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        dataloader = DataLoader(
            self.training_set, batch_size=self.train_batch_size,
            shuffle=self.training_sampler is None,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            sampler=self.training_sampler,
            collate_fn=self._collate_fn,  # type: ignore
            worker_init_fn=self.worker_init  # type: ignore
        )
        assert self.num_train == 0 or len(dataloader) == self.num_train
        return dataloader

    def val_dataloader(self) -> DataLoader:  # type: ignore[override]
        return DataLoader(
            self.validation_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            collate_fn=self._collate_fn,  # type: ignore
            worker_init_fn=self.worker_init  # type: ignore
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[override]
        return DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            collate_fn=self._collate_fn,  # type: ignore
            worker_init_fn=self.worker_init  # type: ignore
        )
