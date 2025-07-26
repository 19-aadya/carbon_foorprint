import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.preprocessing import load_and_preprocess
from utils.graph_utils import build_graph
from models.tabgnn_model import TabGNN



def evaluate_model():
    X, y, _ = load_and_preprocess('data/supply_chain_emission_factors.csv')
    
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("🚨 Dataset is empty after preprocessing! Cannot build graph.")

    X = X.astype(float)
    y = y.astype(float)

    data = build_graph(X, y)

    model = TabGNN(input_dim=1)
    model.load_state_dict(torch.load("results/best_model.pth"))
    model.eval()

    with torch.no_grad():
        preds = model(data).numpy()

    print("✅ Evaluation Results:")
    print("MSE:", mean_squared_error(y, preds))
    print("MAE:", mean_absolute_error(y, preds))
    print("R²:", r2_score(y, preds))
