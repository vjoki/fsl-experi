import sys
from argparse import ArgumentParser
import itertools
import torch
import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from PIL import Image

from snn.omniglot.model import TwinNet

# Quick hack to run a pretrained model on test data.
# Mostly because I don't yet fully trust/understand what pl.test() does for me...
#
# 1. Compares all samples in each class against the same class 5 times in random pairs.
# 2. Compares 300 random pairs from the entire test set.
#
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, default='./data/processed/test/')
    args = vars(parser.parse_args())

    logger = TensorBoardLogger("lightning_logs", name="snn")

    # TODO: Wasn't able to load this without manually specifying batch_size and lr.
    #       Should test again to see if this has changed.
    #checkpoint = torch.load('./models/colab/snn-omniglot-epoch=85.ckpt')
    #print(checkpoint.keys())
    model = TwinNet.load_from_checkpoint(args['model_path'])
    model.eval()
    model.freeze()

    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = TwinNet.add_model_specific_args(parser)
    # args = parser.parse_args()
    # trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    # trainer.test(model, datamodule=OmniglotDataModule())

    foldr = dset.ImageFolder(root=args['test_data_path'])
    correct = 0
    incorrect = 0
    letters = foldr.__len__() // 20

    for i in range(0, foldr.__len__(), 20):
        lbl_correct = 0
        lbl_incorrect = 0

        # Multiple runs to mitigate cases where image1 is set to a particularly hard to match image.
        runs = []
        n_runs = 5
        for _ in range(0, n_runs):
            test_correct = 0
            test_incorrect = 0

            samp = iter(torch.utils.data.SubsetRandomSampler(range(0, 20)))
            (image1, label1) = foldr.__getitem__(i + next(samp))
            image1 = image1.convert('L')
            image1 = transforms.ToTensor()(image1)
            image1.unsqueeze_(0)

            for j in samp:
                (image2, label2) = foldr.__getitem__(i + j)
                image2 = image2.convert('L')
                image2 = transforms.ToTensor()(image2)
                image2.unsqueeze_(0)

                pred = torch.sigmoid(model(image1, image2))
                if label1 == label2 and pred.gt(0.5):
                    test_correct += 1
                elif label1 != label2:
                    print('Label mismatch while iterating pairs, i step is likely incorrect.')
                    sys.exit(1)
                else:
                    test_incorrect += 1
            runs.append((test_correct, test_incorrect))

        # Use the arithmetic mean of test runs.
        lbl_correct = sum([c for (c, _) in runs]) / n_runs
        lbl_incorrect = sum([c for (_, c) in runs]) / n_runs

        if lbl_correct / 20 < 0.6:
            print(label1, ': ', lbl_correct, ' / ', lbl_incorrect, ' = ', lbl_correct / 19)

        correct += lbl_correct
        incorrect += lbl_incorrect

    print('letters: ', letters)
    print('correct: ', correct)
    print('incorrect: ', incorrect)
    print('pct: ', correct / (correct + incorrect))

    # Sample 300 random pairs
    n_samples = 600
    samp = iter(torch.utils.data.RandomSampler(foldr, replacement=False))
    correct = 0

    for i in itertools.islice(samp, n_samples):
        (image1, label1) = foldr.__getitem__(i)
        image1 = image1.convert('L')
        image1 = transforms.ToTensor()(image1)
        image1.unsqueeze_(0)

        (image2, label2) = foldr.__getitem__(next(samp))
        image2 = image2.convert('L')
        image2 = transforms.ToTensor()(image2)
        image2.unsqueeze_(0)

        pred = torch.sigmoid(model(image1, image2))
        if label1 == label2 and pred.gt(0.5):
            correct += 1
        elif label1 != label2 and pred.lt(0.5):
            correct += 1

    print('correct: ', correct, ' ', correct / n_samples)
