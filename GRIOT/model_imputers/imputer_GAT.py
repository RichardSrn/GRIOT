from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
import torch
import torch.nn as nn


class GAT_IMPUTER(torch.nn.Module):
    def __init__(self, d, num_heads=1, dropout=0.5, min_max=None, device=None, concat=True):
        super(GAT_IMPUTER, self).__init__()
        self.device = device
        hidden_dim = max(int(d ** .5), 3)
        self.dropout = dropout
        self.conv1 = GATv2Conv(in_channels=d,
                               out_channels=hidden_dim,
                               dropout=dropout,
                               heads=num_heads,
                               concat=concat).to(self.device)
        self.conv2 = GATv2Conv(in_channels=hidden_dim*(num_heads*concat+1*(not concat)),
                               out_channels=hidden_dim,
                               dropout=dropout,
                               heads=num_heads).to(self.device)
        self.out = torch.nn.Linear(hidden_dim*num_heads, d).to(self.device)
        # # reduce layers parameters floating point precision to 16 bit
        # self.conv1 = self.conv1.to(dtype=torch.float16)
        # self.conv2 = self.conv2.to(dtype=torch.float16)
        # self.out = self.out.to(dtype=torch.float16)
        self.min_max = min_max

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.out(x).sigmoid()
        if self.min_max is not None:
            x = x * (self.min_max[1] - self.min_max[0]) + self.min_max[0]
        return x
