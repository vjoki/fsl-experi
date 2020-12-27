# Fast-ResNet34 (with SE block) implementation from https://github.com/clovaai/voxceleb_trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ResNetBlocks import SEBasicBlock
from .vlad import GhostVLAD, NetVLAD


class ResNetSE(nn.Module):  # pylint: disable=abstract-method
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', **kwargs):
        super().__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))

        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=7, stride=(2, 1), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        # NOTE: Max pool removed.

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))

        # NOTE: Avg pool??? and BN removed.

        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion
        elif self.encoder_type == "ASP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion * 2
        elif self.encoder_type == "NetVLAD":
            vlad_out_dim = 32 * num_filters[3]
            print("ResNet -> VLAD -> FC dimensions: {} -> {} -> {}.".format(num_filters[3], vlad_out_dim, nOut))
            self.vlad = nn.Sequential(
                NetVLAD(num_clusters=32, dim=num_filters[3]),
                nn.Linear(vlad_out_dim, nOut),
                nn.BatchNorm1d(nOut)
            )
        elif self.encoder_type == "GhostVLAD":
            vlad_out_dim = 32 * num_filters[3]
            print("ResNet -> VLAD -> FC dimensions: {} -> {} -> {}.".format(num_filters[3], vlad_out_dim, nOut))
            self.vlad = nn.Sequential(
                GhostVLAD(ghost_clusters=3, vlad_clusters=32, dim=num_filters[3]),
                nn.Linear(vlad_out_dim, nOut),
                nn.BatchNorm1d(nOut)
            )
        else:
            raise ValueError('Undefined encoder')

        # SAP and ASP share this.
        if not self.encoder_type.endswith("VLAD"):
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
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Partial avg pooling?
        x = torch.mean(x, dim=2, keepdim=True)

        if self.encoder_type == "SAP":
            # TODO: Whats this?
            x = x.permute(0, 3, 1, 2).squeeze(-1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)

            x = torch.sum(x * w, dim=1)
        elif self.encoder_type == "ASP":
            x = x.permute(0, 3, 1, 2).squeeze(-1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, rh), 1)
        elif self.encoder_type.endswith("VLAD"):
            x = self.vlad(x)
            x = F.normalize(x, p=2, dim=1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


def MainModel(nOut=256, **kwargs):
    # Number of filters
    num_filters = [16, 32, 64, 128]
    model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model
