from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from snn.omniglot.model import TwinNet, train_and_test

def train():
    parser = ArgumentParser()
    parser.add_argument('--early_stop', type=bool, default=False, help='Enables early stopping')
    parser.add_argument('--early_stop_min_delta', type=int, default=1e-8,
                        help='Minimum change in val_loss quantity to qualify as an improvement')
    parser.add_argument('--early_stop_patience', type=int, default=15,
                        help='# of validation epochs with no improvement after which training will be stopped')
    parser.add_argument('--rng_seed', type=int, default=1, help='RNG seed for reproducibility')
    parser.add_argument('--log_dir', type=str, default='./lightning_logs/', help='Tensorboard log directory')
    parser = TwinNet.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    train_and_test(args)

if __name__ == "__main__":
    train()
