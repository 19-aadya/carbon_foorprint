import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import networkx as nx

# ========== Load and Preprocess ==========
df = pd.read_csv('dataset.csv')  # Replace with actual file
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df = df[df['sector'] == 'Power']
df.sort_values(['country', 'date'], inplace=True)

# Normalize carbon values
scaler = MinMaxScaler()
df['value_scaled'] = scaler.fit_transform(df[['value']])

# Parameters
sequence_length = 5
countries = df['country'].unique()
country_to_idx = {c: i for i, c in enumerate(countries)}

# Node features: past 5-day emissions
features = []
targets = []
country_nodes = []

for country in countries:
    values = df[df['country'] == country]['value_scaled'].values
    if len(values) < sequence_length + 1:
        continue
    for i in range(len(values) - sequence_length):
        X_seq = values[i:i+sequence_length]
        y_seq = values[i+sequence_length]
        features.append(X_seq)
        targets.append(y_seq)
        country_nodes.append(country_to_idx[country])

x = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(targets, dtype=torch.float32)
node_ids = torch.tensor(country_nodes, dtype=torch.long)

# Graph edges: fully connected graph of countries
G = nx.complete_graph(len(countries))
edge_index = torch.tensor(list(G.edges)).t().contiguous()

# Convert to PyG data
data = Data(x=x, edge_index=edge_index, y=y, node_id=node_ids)

# ========== TabGNN Model ==========
class TabGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats=64):
        super().__init__()
        self.gcn1 = GCNConv(in_feats, hidden_feats)
        self.relu = nn.ReLU()
        self.gcn2 = GCNConv(hidden_feats, 1)

    def forward(self, x, edge_index, node_id):
        out = self.gcn1(x, edge_index)
        out = self.relu(out)
        out = self.gcn2(out, edge_index)
        return out.squeeze()

# ========== Train and Evaluate ==========
def train(model, data, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        pred = model(data.x, data.edge_index, data.node_id)
        loss = loss_fn(pred, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index, data.node_id)
    pred = scaler.inverse_transform(pred.numpy().reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(data.y.numpy().reshape(-1, 1)).flatten()
    return {
        'MAE': mean_absolute_error(true, pred),
        'RMSE': mean_squared_error(targets, pred) ** 0.5,
        'R2': r2_score(true, pred)
    }

# ========== Run ==========
model = TabGNN(in_feats=sequence_length)
model = train(model, data)
metrics = evaluate(model, data)

print("\\n🔍 TabGNN Model Performance:")
print(f"MAE = {metrics['MAE']:.4f}")
print(f"RMSE = {metrics['RMSE']:.4f}")
print(f"R2   = {metrics['R2']:.4f}")
