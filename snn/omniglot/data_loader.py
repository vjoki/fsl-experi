import argparse
import os
import shutil
import random
from random import Random
import numpy as np
from PIL import Image
import torch
import pytorch_lightning as pl
import Augmentor
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from typing_extensions import Final
from typing import Optional, Union, List


# TODO: Switch to this once save_hyperparameters is implemented for LightningDataModule
# https://github.com/PyTorchLightning/pytorch-lightning/issues/3769
class OmniglotDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int,
                 num_workers: int, data_path: str, rng_seed: int,
                 way: int, trials: int, num_train: int, **kwargs):
        super().__init__()
        self.batch_size = batch_size

        self._way: Final = way
        self._trials: Final = trials
        self._num_train: Final = num_train

        self._data_path: Final = data_path
        self._rng_seed: Final = rng_seed
        self.save_hyperparameters('batch_size', 'rng_seed', 'way', 'trials', 'num_train')

        # TODO: Does this work right, what about TPUs?
        self._num_workers: Final = num_workers
        self._pin_memory: Final = False
        if torch.cuda.is_available():
            self._num_workers = num_workers
            self._pin_memory = True

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self.training_set = None
        self.validation_set = None
        self.test_set = None

    @staticmethod
    def add_module_specific_args(parser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=1, help='# of workers used by DataLoader')
        parser.add_argument('--trials', type=int, default=320, help='# of 1-shot trials (validation/test)')
        parser.add_argument('--way', type=int, default=20, help='# of ways in 1-shot trials')
        parser.add_argument('--num_train', type=int, default=50000,
                            help='# of pairs in training set (augmented, random pairs)')
        parser.add_argument('--data_path', type=str, default='./data/')
        return parser

    def prepare_data(self):
        # Download and augment datasets if necessary.
        # Should not assing state here.
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


def copy_alphabets(write_dir, alphabets):
    for alphabet in alphabets:
        alpha_dir = os.path.basename(os.path.normpath(alphabet)) + '_'
        for char in os.listdir(alphabet):
            char = os.fsdecode(char)
            dir_name = alpha_dir + char

            val_path = os.path.join(write_dir, dir_name)
            os.makedirs(val_path)

            char_path = os.path.join(alphabet, char)
            for drawer in os.listdir(char_path):
                drawer_path = os.path.join(char_path, drawer)
                shutil.copyfile(
                    drawer_path, os.path.join(
                        val_path, drawer
                    )
                )


# adapted from https://github.com/kevinzakka/one-shot-siamese
class Omniglot(dset.ImageFolder):
    resources = [
        ("https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_background.zip",
         "68d2efa1b9178cc56df9314c21c6e718"),
        ("https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip",
         "6b91aef0f799c5bb55b94e3f2daec811")
    ]

    def __init__(self, data_path: str, mode: str, download: bool = False, seed: int = 0):
        self.processed_path = os.path.join(data_path, 'processed')
        self._rng_seed = seed
        if download is True and not self._processed_check_exists():
            self.raw_path = os.path.join(data_path, 'raw')
            self._download()
            self._process()
        super().__init__(root=os.path.join(self.processed_path, mode))

    def _processed_check_exists(self):
        return (os.path.exists(os.path.join(self.processed_path, 'train')) and
                os.path.exists(os.path.join(self.processed_path, 'valid')) and
                os.path.exists(os.path.join(self.processed_path, 'test')))

    def _raw_check_exists(self):
        return (os.path.exists(os.path.join(self.raw_path, 'images_background')) and
                os.path.exists(os.path.join(self.raw_path, 'images_evaluation')))

    def _download(self):
        if self._processed_check_exists() or self._raw_check_exists():
            return

        os.makedirs(self.raw_path, exist_ok=True)
        for (url, md5) in self.resources:
            filename = url.rpartition('/')[2]
            dset.utils.download_and_extract_archive(url, download_root=self.raw_path, filename=filename, md5=md5)

        bg_path = os.path.join(self.raw_path, 'images_background')
        eval_path = os.path.join(self.raw_path, 'images_evaluation')
        for d in sorted(next(os.walk(eval_path))[1])[:10]:
            shutil.move(os.path.join(eval_path, d), bg_path)

    def _process(self):
        np.random.seed(self._rng_seed)
        os.makedirs(self.processed_path, exist_ok=True)

        back_dir = os.path.join(self.raw_path, 'images_background')
        eval_dir = os.path.join(self.raw_path, 'images_evaluation')

        # get list of all alphabets
        background_alphabets = [os.path.join(back_dir, x) for x in next(os.walk(back_dir))[1]]
        background_alphabets.sort()

        # list of all drawers (1 to 20)
        background_drawers = list(np.arange(1, 21))
        print("There are {} alphabets.".format(len(background_alphabets)))

        # from 40 alphabets, randomly select 30
        train_alphabets = list(np.random.choice(background_alphabets, size=30, replace=False))

        valid_alphabets = [x for x in background_alphabets if x not in train_alphabets]
        test_alphabets = [os.path.join(eval_dir, x) for x in next(os.walk(eval_dir))[1]]

        train_alphabets.sort()
        valid_alphabets.sort()
        test_alphabets.sort()

        copy_alphabets(os.path.join(self.processed_path, 'train'), train_alphabets)
        copy_alphabets(os.path.join(self.processed_path, 'valid'), valid_alphabets)
        copy_alphabets(os.path.join(self.processed_path, 'test'), test_alphabets)


# FIXME: Dataset classes should be refactored to have no state/rng.
#        Randomization should happen in the DataLoader/Sampler.
# from https://github.com/kevinzakka/one-shot-siamese
class TrainSet(Dataset):
    def __init__(self, dataset: Dataset, num_train: int, augment: bool = False):
        super().__init__()
        self.dataset: Final = dataset
        self.num_train: Final = num_train
        self.augment: Final = augment

    def __len__(self):
        return self.num_train

    def __getitem__(self, index):
        rng = random
        image1 = rng.choice(self.dataset.imgs)

        # get image from same class
        label = None
        if index % 2 == 1:
            label = 1.0
            while True:
                image2 = rng.choice(self.dataset.imgs)
                if image1[1] == image2[1]:
                    break
        # get image from different class
        else:
            label = 0.0
            while True:
                image2 = rng.choice(self.dataset.imgs)
                if image1[1] != image2[1]:
                    break
        image1 = Image.open(image1[0])
        image2 = Image.open(image2[0])
        image1 = image1.convert('L')
        image2 = image2.convert('L')

        # apply transformation on the fly
        if self.augment:
            p = Augmentor.Pipeline()
            p.rotate(probability=0.5, max_left_rotation=15, max_right_rotation=15)
            p.random_distortion(
                probability=0.5, grid_width=6, grid_height=6, magnitude=10,
            )
            trans = transforms.Compose([
                p.torch_transform(),
                transforms.ToTensor(),
            ])
        else:
            trans = transforms.ToTensor()

        image1 = trans(image1)
        image2 = transforms.ToTensor()(image2)
        y = torch.from_numpy(np.array([label], dtype=np.float32))
        return (image1, image2, y)


# from https://github.com/kevinzakka/one-shot-siamese
class TestSet(Dataset):
    def __init__(self, dataset: Dataset, trials: int, way: int, seed: int = 0):
        super().__init__()
        self.dataset: Final = dataset
        self.trials: Final = trials
        self.way: Final = way
        self.transform: Final = transforms.ToTensor()
        self.seed: Final = seed

    def __len__(self):
        return (self.trials * self.way)

    def __getitem__(self, index):
        rng = Random(self.seed + index)
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            label = 1.0
            img1 = rng.choice(self.dataset.imgs)
            while True:
                img2 = rng.choice(self.dataset.imgs)
                if img1[1] == img2[1]:
                    break
        # generate image pair from different class
        else:
            label = 0.0
            img1 = rng.choice(self.dataset.imgs)
            while True:
                img2 = rng.choice(self.dataset.imgs)
                if img1[1] != img2[1]:
                    break

        img1 = Image.open(img1[0])
        img2 = Image.open(img2[0])
        img1 = img1.convert('L')
        img2 = img2.convert('L')
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        y = torch.from_numpy(np.array([label], dtype=np.float32))
        return (img1, img2, y)
