from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from snn.librispeech.model import TwinNet

def test():
    parser = ArgumentParser()
    parser.add_argument('--rng_seed', type=int, default=1, help='RNG seed for reproducibility')
    parser.add_argument('--log_dir', type=str, default='./lightning_logs/', help='Tensorboard log directory')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=1, help='# of workers used by DataLoader')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--max_sample_length', type=int, default=None,
                        help='Maximum length of audio samples used, longer samples are clipped to fit.')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.rng_seed)

    logger = TensorBoardLogger(args.log_dir, name='snn')
    trainer = pl.Trainer.from_argparse_args(args, logger=logger,
                                            progress_bar_refresh_rate=20,
                                            deterministic=True)

    model = TwinNet.load_from_checkpoint(args.model_path, stric=False)
    print(model.hparams)

    # Dataloader args.
    model._max_sample_length = args.max_sample_length
    model.batch_size = args.batch_size
    model.num_workers = args.num_workers
    model.data_path = args.data_path

    model.eval()
    model.freeze()

    trainer.test(model=model, ckpt_path=args.model_path)

if __name__ == "__main__":
    test()
