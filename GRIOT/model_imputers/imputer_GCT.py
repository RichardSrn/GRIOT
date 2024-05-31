from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
import torch.nn as nn


class GCT_IMPUTER(torch.nn.Module):
    """
    Graph Convolutional Transformer (GCT) imputer
    """

    def __init__(self, d, num_heads=1, dropout=0.5, min_max=None, device=None, concat=True):
        super(GCT_IMPUTER, self).__init__()
        self.device = device
        hidden_dim = max(int(d ** .5), 3)
        self.conv1 = TransformerConv(in_channels=d,
                                     out_channels=hidden_dim,
                                     heads=num_heads,
                                     dropout=dropout,
                                     concat=concat).to(self.device)
        self.conv2 = TransformerConv(in_channels=hidden_dim*(num_heads*concat+1*(not concat)),
                                     out_channels=hidden_dim,
                                     heads=num_heads,
                                     dropout=dropout).to(self.device)
        self.out = torch.nn.Linear(hidden_dim*num_heads, d).to(self.device)
        self.min_max = min_max

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.sigmoid(self.out(x))
        if self.min_max is not None:
            x = x * (self.min_max[1] - self.min_max[0]) + self.min_max[0]
        return x
