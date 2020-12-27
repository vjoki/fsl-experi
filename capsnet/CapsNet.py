# CapsNet implementation from https://github.com/adambielski/CapsNet-pytorch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(x):
    norm = x.norm(dim=2)
    norm2 = norm.pow(2)
    x = x * (norm / (norm2 + 1)).view(x.size(0), x.size(1), 1)
    return x


class AgreementRouting(nn.Module):  # pylint: disable=abstract-method
    def __init__(self, input_caps, output_caps, n_iterations):
        super().__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b, dim=1)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for _ in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps), dim=1).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)

        return v


class CapsLayer(nn.Module):  # pylint: disable=abstract-method
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super().__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        v = self.routing_module(u_predict)
        return v


class PrimaryCapsLayer(nn.Module):  # pylint: disable=abstract-method
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out)
        return out


class CapsNet(nn.Module):  # pylint: disable=abstract-method
    def __init__(self, routing_iterations, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primaryCaps = PrimaryCapsLayer(256, 32, 8, kernel_size=9, stride=2)  # outputs 6*6
        self.num_primaryCaps = 32 * 6 * 6
        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(self.num_primaryCaps, 8, n_classes, 16, routing_module)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.primaryCaps(x)
        x = self.digitCaps(x)
        probs = x.pow(2).sum(dim=2).sqrt()
        return x, probs


class CapsNetWithoutPrimaryCaps(nn.Module):  # pylint: disable=abstract-method
    def __init__(self, routing_iterations, input_caps, input_dim, output_caps, output_dim):
        super().__init__()
        routing_module = AgreementRouting(input_caps, output_caps, routing_iterations)
        self.digitCaps = CapsLayer(input_caps, input_dim, output_caps, output_dim, routing_module)

    def forward(self, input):
        x = self.digitCaps(input)
        probs = x.pow(2).sum(dim=2).sqrt()
        return x, probs
