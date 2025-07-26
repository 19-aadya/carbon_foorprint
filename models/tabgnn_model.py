# models/tabgnn_model.py
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TabGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):  # Adding hidden_dim as a parameter with a default value
        super(TabGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # Use hidden_dim here
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)  # You can adjust the architecture here as needed
        self.fc = nn.Linear(hidden_dim // 2, 1)  # Similarly, adjust the final layer as needed

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x.view(-1)
