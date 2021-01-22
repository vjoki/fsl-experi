import argparse
from typing import List
import pytorch_lightning as pl

from snn.librispeech.model import BaseNet, SNN, SNNCapsNet, SNNAngularProto, SNNSoftmaxProto
from snn.librispeech.datamodule import LibriSpeechDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def train_and_test(args: argparse.Namespace):
    dict_args = vars(args)
    seed = args.rng_seed
    log_dir = args.log_dir
    early_stop = args.early_stop
    early_stop_min_delta = args.early_stop_min_delta
    early_stop_patience = args.early_stop_patience
    checkpoint_dir = args.checkpoint_dir

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
        monitor='val_loss' if args.fast_dev_run else 'val_eer', mode='min',
        filepath=checkpoint_dir + args.model + '-{epoch}-{val_loss:.2f}-{val_eer:.2f}',
        save_top_k=3
    )

    logger = TensorBoardLogger(log_dir, name=args.model, log_graph=True, default_hp_metric=False)
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, progress_bar_refresh_rate=20,
                                            # Gives rise to NaN with binary_cross_entropy_with_logits?
                                            #precision=16,
                                            deterministic=True, auto_lr_find=True,
                                            checkpoint_callback=checkpoint_callback,
                                            callbacks=callbacks)

    model: BaseNet
    if args.model == 'snn':
        model = SNN(**dict_args)
        datamodule = LibriSpeechDataModule(train_set_type='pair', **dict_args)
    elif args.model == 'snn-capsnet':
        model = SNNCapsNet(**dict_args)
        datamodule = LibriSpeechDataModule(train_set_type='pair', **dict_args)
    elif args.model == 'snn-angularproto':
        model = SNNAngularProto(**dict_args)
        datamodule = LibriSpeechDataModule(train_set_type='nshotkway', **dict_args)
    elif args.model == 'snn-softmaxproto':
        model = SNNSoftmaxProto(**dict_args)
        datamodule = LibriSpeechDataModule(train_set_type='nshotkway', **dict_args)

    # Tune learning rate.
    trainer.tune(model, datamodule=datamodule)
    logger.log_hyperparams(params=model.hparams)

    # Train model.
    trainer.fit(model, datamodule=datamodule)
    print('Best model saved to: ', checkpoint_callback.best_model_path)
    trainer.save_checkpoint(checkpoint_dir + args.model + '-last.ckpt')

    # Test using best checkpoint.
    trainer.test(datamodule=datamodule)


def train():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    general = parser.add_argument_group('General')
    general.add_argument('--model', type=str.lower, default='snn',
                         choices=['snn', 'snn-capsnet', 'snn-angularproto', 'snn-softmaxproto'],
                         help='Choose the model to train.')

    general.add_argument('--early_stop', action='store_true', default=False, help='Enable early stopping')
    general.add_argument('--early_stop_min_delta', type=int, default=1e-8,
                         help='Minimum change in val_loss quantity to qualify as an improvement')
    general.add_argument('--early_stop_patience', type=int, default=15,
                         help='# of validation epochs with no improvement after which training will be stopped')
    general.add_argument('--log_dir', type=str, default='./lightning_logs/', help='Tensorboard log directory')
    general.add_argument('--checkpoint_dir', type=str, default='./checkpoints/', help='Model checkpoint directory.')

    parser = BaseNet.add_model_specific_args(parser)
    parser = LibriSpeechDataModule.add_dataset_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    train_and_test(args)


if __name__ == "__main__":
    train()
