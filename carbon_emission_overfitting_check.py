
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# ========== Load and Preprocess Data ==========
df = pd.read_csv('dataset.csv')  # Replace with your file
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df = df[df['sector'] == 'Power']
df.sort_values(['country', 'date'], inplace=True)

scaler = MinMaxScaler()
df['value_scaled'] = scaler.fit_transform(df[['value']])
sequence_length = 5
X, y = [], []

for country in df['country'].unique():
    values = df[df['country'] == country]['value_scaled'].values
    for i in range(len(values) - sequence_length):
        X.append(values[i:i+sequence_length])
        y.append(values[i+sequence_length])
X, y = np.array(X), np.array(y)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ========== Define Model ==========
class RNNModel(nn.Module):
    def __init__(self, model_type='LSTM'):
        super().__init__()
        hidden_size = 32
        if model_type == 'BiLSTM':
            self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden_size * 2, 1)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden_size, 1)
        else:
            self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze()

# ========== Train, Evaluate and Visualize ==========
def train_model(model, train_loader, test_loader, model_name, epochs=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        model.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                epoch_test_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))
        test_losses.append(epoch_test_loss / len(test_loader))

    # Plot loss curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title(f'{model_name} Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(f'{model_name}_loss.png')

    return model

def evaluate_model(model, test_loader, model_name):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            preds.extend(pred.numpy())
            targets.extend(yb.numpy())

    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    targets = scaler.inverse_transform(np.array(targets).reshape(-1,1)).flatten()

    # Plot predictions
    plt.figure()
    plt.plot(targets, label='True')
    plt.plot(preds, label='Predicted')
    plt.title(f'{model_name}: True vs Predicted')
    plt.legend()
    plt.savefig(f'{model_name}_prediction.png')

    return {
        'MAE': mean_absolute_error(targets, preds),
        'RMSE': mean_squared_error(targets, preds) ** 0.5,
        'R2': r2_score(targets, preds)
    }

# ========== Cross Validation and Results ==========
results = {}
tscv = TimeSeriesSplit(n_splits=3)

for model_name in ['LSTM', 'BiLSTM', 'GRU']:
    all_metrics = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        train_ds = TimeSeriesDataset(X[train_idx], y[train_idx])
        test_ds = TimeSeriesDataset(X[test_idx], y[test_idx])
        train_loader = DataLoader(train_ds, batch_size=16)
        test_loader = DataLoader(test_ds, batch_size=16)

        model = RNNModel(model_type=model_name)
        model = train_model(model, train_loader, test_loader, model_name)
        metrics = evaluate_model(model, test_loader, model_name)
        all_metrics.append(metrics)

    avg_metrics = {
        'MAE': np.mean([m['MAE'] for m in all_metrics]),
        'RMSE': np.mean([m['RMSE'] for m in all_metrics]),
        'R2': np.mean([m['R2'] for m in all_metrics])
    }
    results[model_name] = avg_metrics

# Final Results
print("\nModel Comparison after Regularization and Cross-Validation:")
for name, metric in results.items():
    print(f"{name}: MAE={metric['MAE']:.4f}, RMSE={metric['RMSE']:.4f}, R2={metric['R2']:.4f}")
