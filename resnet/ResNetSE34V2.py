# Thin-ResNet34 (with SE block) implementation from https://github.com/clovaai/voxceleb_trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet.ResNetBlocks import SEBasicBlock
from resnet.vlad import GhostVLAD, NetVLAD


class ResNetSE(nn.Module):  # pylint: disable=abstract-method
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', n_mels=40, **kwargs):
        super().__init__()

        print('Embedding size is %d, encoder %s.' % (nOut, encoder_type))

        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels = n_mels

        # NOTE: Kernel, stride and padding differ from original ResNet.
        # Also order of ReLU and BN has been swapped.
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        # NOTE: Max pool removed.

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(2, 2))

        # NOTE: Avg pool and BN removed.

        outmap_size = int(self.n_mels/8)

        # SAP and ASP share this.
        if not self.encoder_type.endswith("VLAD"):
            self.attention = nn.Sequential(
                nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
                nn.Softmax(dim=2),
            )

        out_dim: int
        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        elif self.encoder_type == "NetVLAD":
            out_dim = 32 * num_filters[3]
            print("ResNet -> VLAD dimensions: {} -> {}.".format(num_filters[3], out_dim))
            self.vlad = NetVLAD(num_clusters=32, dim=num_filters[3])
        elif self.encoder_type == "GhostVLAD":
            out_dim = 32 * num_filters[3]
            print("ResNet -> VLAD dimensions: {} -> {}.".format(num_filters[3], out_dim))
            self.vlad = GhostVLAD(ghost_clusters=3, vlad_clusters=32, dim=num_filters[3])
        else:
            raise ValueError('Undefined encoder')

        print("ResNet -> FC dimensions: {} -> {}.".format(out_dim, nOut))
        self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def new_parameter(*size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # SAP and ASP share this.
        if not self.encoder_type.endswith("VLAD"):
            x = x.reshape(x.size()[0], -1, x.size()[-1])
            w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, sg), 1)
        elif self.encoder_type.endswith("VLAD"):
            # *VLAD -> FC -> L2 norm
            x = self.vlad(x)
            x = self.fc(x)
            x = F.normalize(x, p=2, dim=1)

        # SAP and ASP share this.
        if not self.encoder_type.endswith("VLAD"):
            x = x.view(x.size()[0], -1)
            x = self.fc(x)

        return x


def MainModel(nOut=256, **kwargs):
    # Number of filters
    num_filters = [32, 64, 128, 256]
    model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model
