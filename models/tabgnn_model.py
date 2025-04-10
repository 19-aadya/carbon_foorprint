# models/tabgnn_model.py
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TabGNN(nn.Module):
    def __init__(self, input_dim):
        super(TabGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, 8)
        self.fc = nn.Linear(8, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x.view(-1)
