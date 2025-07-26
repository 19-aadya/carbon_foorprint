
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ========== Load and Preprocess Data ==========
df = pd.read_csv('dataset.csv')  # Replace with your file
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df = df[df['sector'] == 'Power']
df.sort_values(['country', 'date'], inplace=True)

# Normalize target value (carbon footprint emission)
scaler = MinMaxScaler()
df['value_scaled'] = scaler.fit_transform(df[['value']])

# Create time series sequences
sequence_length = 5
X, y = [], []
for country in df['country'].unique():
    values = df[df['country'] == country]['value_scaled'].values
    for i in range(len(values) - sequence_length):
        X.append(values[i:i+sequence_length])
        y.append(values[i+sequence_length])
X, y = np.array(X), np.array(y)

# Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# Split
dataset = TimeSeriesDataset(X, y)
train_size = int(0.8 * len(dataset))
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
train_loader = DataLoader(train_data, batch_size=16)
test_loader = DataLoader(test_data, batch_size=16)

# ========== Define Models ==========
class RNNModel(nn.Module):
    def __init__(self, model_type='LSTM'):
        super().__init__()
        if model_type == 'BiLSTM':
            self.rnn = nn.LSTM(input_size=1, hidden_size=32, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(64, 1)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_size=1, hidden_size=32, batch_first=True)
            self.fc = nn.Linear(32, 1)
        else:
            self.rnn = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
            self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()

# ========== Train and Evaluate ==========
def train(model, loader, epochs=30):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()

def evaluate(model, loader):
    model.eval(); preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            preds += pred.numpy().tolist()
            targets += yb.numpy().tolist()
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    targets = scaler.inverse_transform(np.array(targets).reshape(-1,1)).flatten()
    return {
        'MAE': mean_absolute_error(targets, preds),
        'RMSE': mean_squared_error(targets, preds) ** 0.5,
        'R2': r2_score(targets, preds)
    }

# ========== Run Models ==========
results = {}
for model_name in ['LSTM', 'BiLSTM', 'GRU']:
    model = RNNModel(model_type=model_name)
    train(model, train_loader)
    results[model_name] = evaluate(model, test_loader)

# Print results
print("\nModel Performance on Carbon Emission Forecasting (Power sector):")
for model_name, metrics in results.items():
    print(f"{model_name}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
