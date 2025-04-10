
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

def build_graph(X, y):
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    # Create fully connected graph (or custom similarity-based if needed)
    num_nodes = x.size(0)
    adj = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)  # no self-loops initially

    edge_index, _ = dense_to_sparse(adj)

    # Safety: if empty, add self-loops
    if edge_index.size(1) == 0:
        print("No edges found. Adding self-loops.")
        edge_index = torch.arange(num_nodes).repeat(2, 1)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data
