from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from snn.omniglot.model import TwinNet, train_and_test

def train():
    parser = ArgumentParser()
    parser.add_argument('--rng_seed', type=int, default=1, help='RNG seed for reproducibility')
    parser.add_argument('--log_dir', type=str, default='./lightning_logs/', help='Tensorboard log directory')
    parser = TwinNet.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    train_and_test(args)

if __name__ == "__main__":
    train()
