from torch_geometric.nn import GCNConv, conv, global_add_pool, GINConv
import torch.nn.functional as F
import torch
import torch.nn as nn


class GCN_IMPUTER(torch.nn.Module):
    def __init__(self, d, dropout=0.5, min_max=None, device=None):
        super(GCN_IMPUTER, self).__init__()
        self.dropout = dropout
        self.device = device
        self.conv1 = GCNConv(in_channels=d,
                             out_channels=int(d ** .5)).to(self.device)
        self.conv2 = GCNConv(in_channels=int(d ** .5),
                             out_channels=int(d ** .5)).to(self.device)
        self.out = torch.nn.Linear(int(d ** .5), d).to(self.device)
        self.min_max = min_max

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.sigmoid(self.out(x))
        if self.min_max is not None:
            x = x * (self.min_max[1] - self.min_max[0]) + self.min_max[0]
        return x
