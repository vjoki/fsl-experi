# Implementation adapted from https://github.com/lyakaap/NetVLAD-pytorch/,
# https://github.com/Nanne/pytorch-NetVlad/ and https://github.com/sitzikbs/netVLAD/
import torch
import torch.nn as nn
import torch.nn.functional as F


# GhostVLAD, variant of NetVLAD: https://arxiv.org/abs/1810.09951
class GhostVLAD(nn.Module):  # pylint: disable=abstract-method
    def __init__(self, vlad_clusters=8, ghost_clusters=0, alpha=100.0, dim=128, normalize_input=True):
        super().__init__()
        self.K = vlad_clusters
        self.G = ghost_clusters
        self.Df = dim
        self.alpha = alpha
        self.normalize_input = normalize_input

        # Random initialization.
        self.conv = nn.Conv2d(self.Df, self.K + self.G, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(self.K, self.Df))

        # TODO: Ghost clusters should be initialized from degraded input using k-means.
        # self.conv.weight = nn.Parameter((2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
        # self.conv.bias = nn.Parameter(-self.alpha * torch.linalg.norm(self.centroids, dim=1))

    def forward(self, x):
        N, C = x.shape[:2]
        assert C == self.Df, "feature dim not correct"

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        soft_assign = self.conv(x).view(N, self.K + self.G, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        # Remove ghost assign.
        soft_assign = soft_assign[:, :self.K, :]

        x_flatten = x.view(N, C, -1)

        vlad = torch.zeros([N, self.K, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.K):
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C+1, :].unsqueeze(2)
            vlad[:, C:C+1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(x.size(0), -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


# https://arxiv.org/abs/1511.07247
class NetVLAD(GhostVLAD):  # pylint: disable=abstract-method
    def __init__(self, num_clusters=8, alpha=100.0, dim=128, normalize_input=True):
        super().__init__(ghost_clusters=0, vlad_clusters=num_clusters, alpha=alpha, dim=dim,
                         normalize_input=normalize_input)
