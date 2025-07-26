import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

def build_graph(X, y):
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    num_nodes = x.size(0)

    if num_nodes == 0:
        raise ValueError("No data provided: X has 0 samples.")

    # Create adjacency matrix for a fully connected graph without self-loops
    adj = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)

    print(f"Adjacency Matrix:\n{adj}")  # Now adj will definitely exist

    edge_index, _ = dense_to_sparse(adj)

    # If no edges are found, fallback to self-loops
    if edge_index.size(1) == 0:
        print("No edges found. Adding self-loops.")
        edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])

    data = Data(x=x, edge_index=edge_index, y=y)
    return data
