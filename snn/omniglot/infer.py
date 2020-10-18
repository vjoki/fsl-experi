from argparse import ArgumentParser
import itertools
import torch
import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from PIL import Image

from snn.omniglot.model import TwinNet

# Runs inference on input images using existing model.
if __name__ == "__main__":
    logger = TensorBoardLogger("lightning_logs", name="snn")
    parser = ArgumentParser()
    parser.add_argument('--img1', type=str, default='test.png')
    parser.add_argument('--img2', type=str, default='test2.png')
    args = vars(parser.parse_args())

    checkpoint = torch.load('./models/colab/latest.ckpt')
    model = TwinNet.load_from_checkpoint('./models/colab/latest.ckpt', batch_size=128, learning_rate=1e-3)
    model.eval()
    model.freeze()

    image1 = Image.open(args['img1'])
    image1 = image1.convert('L')
    image1 = transforms.ToTensor()(image1)
    image1.unsqueeze_(0)

    image2 = Image.open(args['img2'])
    image2 = image2.convert('L')
    image2 = transforms.ToTensor()(image2)
    image2.unsqueeze_(0)

    print(torch.sigmoid(model(image1, image2)))
