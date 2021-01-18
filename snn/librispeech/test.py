import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from snn.librispeech.dataset import PairDatasetFromList
from snn.librispeech.datamodule import LibriSpeechDataModule
from snn.librispeech.model import BaseNet, SNN, SNNCapsNet, SNNAngularProto, SNNSoftmaxProto


def test():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    general = parser.add_argument_group('General')
    general.add_argument('--log_dir', type=str, default='./lightning_logs/', help='Tensorboard log directory')
    general.add_argument('--model', type=str.lower, default='SNN',
                         choices=['snn', 'snn-capsnet', 'snn-angularproto', 'snn-softmaxproto'],
                         help='Type of the model to load.')
    general.add_argument('--model_path', type=str, required=True)
    general.add_argument('--test_list', type=str, default=None, help='File containing list of test pairs and labels')
    general.add_argument('--test_path', type=str, default='./',
                         help='Test data path, only applies when using --test_list.')
    parser = BaseNet.add_model_specific_args(parser)
    parser = LibriSpeechDataModule.add_dataset_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.rng_seed)

    logger = TensorBoardLogger(args.log_dir, name='snn')
    trainer = pl.Trainer.from_argparse_args(args, logger=logger,
                                            progress_bar_refresh_rate=20,
                                            deterministic=True)

    model: BaseNet
    if args.model == 'snn':
        model = SNN
    elif args.model == 'snn-capsnet':
        model = SNNCapsNet
    elif args.model == 'snn-angularproto':
        model = SNNAngularProto
    elif args.model == 'snn-softmaxproto':
        model = SNNSoftmaxProto
    model = model.load_from_checkpoint(args.model_path, strict=False,
                                       # Filter defaulted flags so that we don't needlessly override hparams.
                                       **{k: v for k, v in vars(args).items() if v != parser.get_default(k)})
    print(model.hparams)

    model.eval()
    model.freeze()

    test_dataloader = None
    if args.test_list:
        test_dataloader = DataLoader(
            PairDatasetFromList(args.test_list, args.test_path,
                                max_sample_length=args.max_sample_length),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
        )

    trainer.test(model=model, ckpt_path=args.model_path, test_dataloaders=test_dataloader)


if __name__ == "__main__":
    test()
