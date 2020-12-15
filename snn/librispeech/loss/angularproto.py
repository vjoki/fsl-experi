# Adapted from https://github.com/clovaai/voxceleb_trainer/
import torch
import torch.nn as nn
import torch.nn.functional as F


# class AngularPrototypicalLoss(nn.Module):
#     def __init__(self, init_scale=10.0, init_bias=-5.0):
#         super().__init__()
#         self.w = nn.Parameter(torch.as_tensor(init_scale))
#         self.b = nn.Parameter(torch.as_tensor(init_bias))
#         self.criterion = torch.nn.CrossEntropyLoss()

#     def forward(self, query, support, label=None):
#         # B = batch, W = way = speaker, S = shot, N = features
#         # query shape is BxN.
#         # support shape is BxWxSxN, where WxS signifies the mini-batch.
#         assert support.size(1) >= 2
#         #print(support.shape)
#         #print(label.shape)

#         # Average support set shots.
#         # out_anchor = prototype = centroid
#         centroid = torch.mean(support, 2)  # BxWxN
#         #print(centroid.shape, query.shape)

#         # Cosine similarity needs both inputs to be the same shape.
#         query = query.unsqueeze(-2).repeat(1, centroid.size(1), 1)  # BxWxN

#         # Feature dimension (N) wise cosine similiarity.
#         cos_sim_matrix = F.cosine_similarity(query, centroid, dim=2)  # BxW

#         # TODO: Why is w clamped?
#         torch.clamp(self.w, 1e-6)
#         cos_sim_matrix = self.w * cos_sim_matrix + self.b  # BxW

#         # print(cos_sim_matrix.shape, label.shape)
#         # losses = []
#         # for i in range(0, label.size(1)):
#         #     losses.append(self.criterion(cos_sim_matrix, label[:, i]))
#         # nloss = torch.mean(torch.stack(losses))

#         # We assume that the first support set is the query speaker, thus target class is always 0.
#         #label = torch.zeros(centroid.size(0), dtype=torch.long, device=query.device)
#         if label is None:
#             label = torch.arange(0, centroid.size(0), dtype=torch.long, device=query.device)
#             #label = torch.zeros(centroid.size(0), dtype=torch.long, device=query.device)
#         #label[0] = 1
#         nloss = self.criterion(cos_sim_matrix, label)
#         #print(nloss, nloss.shape, cos_sim_matrix.shape, label.shape)

#         return nloss, cos_sim_matrix, label


class AngularPrototypicalLoss(nn.Module):
    def __init__(self, init_scale=10.0, init_bias=-5.0):
        super().__init__()
        self.w = nn.Parameter(torch.as_tensor(init_scale))
        self.b = nn.Parameter(torch.as_tensor(init_bias))
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, support, label=None):
        # B = batch = way = speakers, S = shots, N = features
        # support shape is BxSxN.
        assert support.size(0) >= 2
        #print(support.shape)
        #print(label.shape)

        # Average support set shots.
        # out_anchor = prototype = centroid
        centroid = torch.mean(support[1:, :, :], 1)  # BxN
        #print(centroid.shape, query.shape)

        # Cosine similarity needs both inputs to be the same shape.
        query = support[0, :, :].unsqueeze(-2)  # BxN

        # Feature dimension (N) wise cosine similiarity.
        cos_sim_matrix = F.cosine_similarity(query, centroid, dim=0)  # B

        # TODO: Why is w clamped?
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = self.w * cos_sim_matrix + self.b  # B

        # print(cos_sim_matrix.shape, label.shape)
        # losses = []
        # for i in range(0, label.size(1)):
        #     losses.append(self.criterion(cos_sim_matrix, label[:, i]))
        # nloss = torch.mean(torch.stack(losses))

        # We assume that the first support set is the query speaker, thus target class is always 0.
        #label = torch.zeros(centroid.size(0), dtype=torch.long, device=query.device)
        if label is None:
            label = torch.arange(0, centroid.size(0), dtype=torch.long, device=query.device)
            #label = torch.zeros(centroid.size(0), dtype=torch.long, device=query.device)
        #label[0] = 1
        nloss = self.criterion(cos_sim_matrix, label)
        #print(nloss, nloss.shape, cos_sim_matrix.shape, label.shape)

        return nloss, cos_sim_matrix, label
