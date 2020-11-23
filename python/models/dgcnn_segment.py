import torch
import torch.nn as nn
import torch.nn.functional as F
from caloGraphNN import *
from ops.ties import EdgeConv

class DgcnnSegment(torch.nn.Module):
    def __init__(self, input_size=128, num_neighbors=10):
        super(DgcnnSegment, self).__init__()
        self.bn = nn.BatchNorm1d(input_size, momentum=0.01)
        self.preprocess = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU()
            )

        self.edge_convs = nn.ModuleList([
            EdgeConv(num_neighbors=10, input_size=64, mlp_layers=[128, 64, 64]),
            EdgeConv(num_neighbors=10, input_size=64, mlp_layers=[128, 64, 64]),
            EdgeConv(num_neighbors=10, input_size=64, mlp_layers=[128, 64, 64]),
            EdgeConv(num_neighbors=10, input_size=64, mlp_layers=[128, 64, 64])
        ])

        self.mlp2 = nn.Sequential(
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Linear(192, 192),
            nn.ReLU(),
        )

    def forward(self, feat):
        shape = feat.shape
        feat = self.bn(feat.view(-1, feat.shape[-1])).view(*feat.shape)
        feat = self.preprocess(feat)

        feat = self.edge_convs[0](feat)
        feat1 = self.edge_convs[1](feat)
        feat2 = self.edge_convs[2](feat)
        feat3 = self.edge_convs[3](feat)

        feat = torch.cat([feat1, feat2, feat3], axis=-1)
        feat = self.mlp2(feat)
        return feat
