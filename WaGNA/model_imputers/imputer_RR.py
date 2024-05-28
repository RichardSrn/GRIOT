from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch

class GNN_RR(torch.nn.Module):
    def __init__(self, d, device):
        super(GNN_RR, self).__init__()
        self.device = device
        self.conv1 = GCNConv(d-1, 1).to(self.device)
        # self.conv1 = GCNConv(d-1, int(d**.5)).to(device)
        # self.conv2 = GCNConv(int(d**.5), 1).to(device)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x = x.relu()
        # x = F.dropout(x, p=.5, training=self.training)
        # x = self.conv2(x, edge_index)
        # x = x.sigmoid()
        return x