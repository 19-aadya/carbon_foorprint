import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import kneighbors_graph
from torch.nn import Linear, ReLU, MSELoss
from torch.optim import Adam
from torch_geometric.nn import GCNConv
import numpy as np

# Load your dataset
df = pd.read_csv("data/supply_chain_emission_factors.csv")
df = df.rename(columns={
    '2017 NAICS Title': 'Sector',
    'Supply Chain Emission Factors with Margins': 'EmissionFactor'
})
df = df[['Sector', 'EmissionFactor']]  # keep relevant columns
df.dropna(inplace=True)  # remove rows with missing values

# Encode the 'Sector' text into numbers
df['Sector_encoded'] = df['Sector'].astype('category').cat.codes
X = df[['Sector_encoded']].values
y = df['EmissionFactor'].values

# Fully connected graph
def build_fully_connected_graph(X, y):
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    num_nodes = x.size(0)
    adj = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)
    edge_index, _ = dense_to_sparse(adj)
    return Data(x=x, edge_index=edge_index, y=y)

# KNN graph
def build_knn_graph(X, y, k=10):
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    knn_graph = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.vstack(knn_graph.nonzero()), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

# GCN Model
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.relu = ReLU()
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze()

# Training function
def train_and_evaluate(data):
    model = GCN()
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_fn = MSELoss()
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = model(data)
    mse = mean_squared_error(data.y.numpy(), preds.numpy())
    return model, mse

# Build both graphs and compare
data_fc = build_fully_connected_graph(X, y)
data_knn = build_knn_graph(X, y, k=10)

# Train and evaluate both models
model_fc, mse_fc = train_and_evaluate(data_fc)
model_knn, mse_knn = train_and_evaluate(data_knn)

# Calculate R² and MAE for Fully Connected Graph
preds_fc = model_fc(data_fc).detach().numpy()
mae_fc = mean_absolute_error(y, preds_fc)
r2_fc = r2_score(y, preds_fc)

# Calculate R² and MAE for KNN Graph
preds_knn = model_knn(data_knn).detach().numpy()
mae_knn = mean_absolute_error(y, preds_knn)
r2_knn = r2_score(y, preds_knn)

# Print results
print("📊 Fully Connected Graph MSE:", mse_fc)
print("📊 Fully Connected Graph MAE:", mae_fc)
print("📊 Fully Connected Graph R²:", r2_fc)

print("\n🔗 KNN Graph (k=10) MSE:", mse_knn)
print("🔗 KNN Graph (k=10) MAE:", mae_knn)
print("🔗 KNN Graph (k=10) R²:", r2_knn)
