# Adapted from https://github.com/clovaai/voxceleb_trainer/
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet.utils import accuracy


class AngularPrototypicalLoss(nn.Module):
    def __init__(self, init_scale=10.0, init_bias=-5.0, **kwargs):
        super().__init__()
        self.w = nn.Parameter(torch.as_tensor(init_scale))
        self.b = nn.Parameter(torch.as_tensor(init_bias))
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, support, label=None):
        # B = batch = way = speakers, S = shots, N = features
        # support shape is BxWxSxN.
        n_ways = support.size(1)
        n_shots = support.size(2)

        # Our cosine similarity compares the first shots of each way to the average of the rest of the shots.
        # Thus n_shots must be >=2.
        assert n_shots >= 2

        # First shots of each way.
        query = support[:, :, 0, :]  # BxWxN

        # Average of the rest of the shots of each way.
        # out_anchor = prototype = centroid
        centroid = torch.mean(support[:, :, 1:, :], 2)  # BxWxN

        # Feature dimension (N) wise cosine similiarity.
        # Ex. for
        # support = tensor([[[[ 1.2535,  0.5587],
        #                     [ 1.2535,  0.5587]],
        #                    [[-0.5054,  1.0834],
        #                      [-0.5054,  1.0834]],
        #                    [[ 1.3112,  0.8206],
        #                      [ 1.3112,  0.8206]]]])
        #
        # query = tensor([[[ 1.2535,  0.5587],
        #                  [-0.5054,  1.0834],
        #                  [ 1.3112,  0.8206]]])
        # centroid = tensor([[[ 1.2535,  0.5587],
        #                     [-0.5054,  1.0834],
        #                     [ 1.3112,  0.8206]]])
        #
        # cos_sim_matrix = tensor([[ 1.0000, -0.0172,  0.9902],
        #                          [-0.0172,  1.0000,  0.1224],
        #                          [ 0.9902,  0.1224,  1.0000]])
        #
        # label = tensor([0, 1, 2])
        # loss = tensor(0.7694)
        #
        cos_sim_matrix = F.cosine_similarity(
            # BxWxN -> B*WxNx1
            query.reshape(support.size(0) * n_ways, -1, 1),
            # BxWxN -> B*WxNx1 -> 1xNxB*W
            centroid.reshape(support.size(0) * n_ways, -1, 1).transpose(0, 2),
            dim=1
        )  # B*WxB*W

        # TODO: Why is w clamped?
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = self.w * cos_sim_matrix + self.b  # B*WxB*W

        # We assume that the first support set is the query speaker, thus target class is always 0.
        if label is None:
            label = torch.arange(0, n_ways, dtype=torch.long, device=support.device)
        nloss = self.criterion(cos_sim_matrix, label)
        acc = accuracy(cos_sim_matrix.detach(), label.detach(), topk=(1,))[0]

        return nloss, acc
