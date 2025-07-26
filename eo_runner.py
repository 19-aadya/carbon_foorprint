import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch

from models.tabgnn_model import TabGNN
from models.equilibrium_optimizer import EO_feature_selector, EO_hyperparam_tune
from utils.graph_utils import build_graph
from utils.preprocessing import load_and_preprocess

# Load and preprocess the dataset
X, y, df = load_and_preprocess("data/supply_chain_emission_factors.csv")

# ---------- EO for Feature Selection ----------
def feature_score(X_sub, y):
    """Evaluate feature subset by training a simple linear model and returning MSE."""
    X_train, X_test, y_train, y_test = train_test_split(X_sub, y, test_size=0.2, random_state=42)
    model = torch.nn.Linear(X_train.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    for _ in range(50):
        pred = model(torch.tensor(X_train, dtype=torch.float32))
        loss = loss_fn(pred.squeeze(), torch.tensor(y_train, dtype=torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32))
    return mean_squared_error(y_test, pred.squeeze().numpy())

# Apply EO for feature selection
selected_mask = EO_feature_selector(X, y, feature_score)
X_selected = X[:, selected_mask]

# ---------- EO for Hyperparameter Tuning ----------
def model_eval(hidden_dim, lr):
    """Evaluate TabGNN performance with given hyperparameters."""
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    data = build_graph(X_train, y_train)

    model = TabGNN(input_dim=X_selected.shape[1], hidden_dim=int(hidden_dim))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = torch.nn.MSELoss()

    for _ in range(50):
        model.train()
        pred = model(data)
        loss = loss_fn(pred.squeeze(), torch.tensor(y_train, dtype=torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(build_graph(X_test, y_test))

    return mean_squared_error(y_test, pred.squeeze().numpy())

# Run EO for hyperparameter tuning
best_hyperparams = EO_hyperparam_tune([(8, 64), (0.001, 0.05)], model_eval)
print("Best Hidden Dim & LR:", best_hyperparams)
