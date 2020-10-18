from argparse import ArgumentParser
import torch
from torchvision import transforms
from PIL import Image

from snn.omniglot.model import TwinNet


# Runs inference on input images using existing model.
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--img1', type=str, default='test.png')
    parser.add_argument('--img2', type=str, default='test2.png')
    args = vars(parser.parse_args())

    model = TwinNet.load_from_checkpoint(args['model_path'])
    print(model.hparams)
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
