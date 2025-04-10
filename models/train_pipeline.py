# models/train_pipeline.py
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocessing import load_and_preprocess
from utils.graph_utils import build_graph
from models.tabgnn_model import TabGNN
from models.equilibrium_optimizer import EquilibriumOptimizer
import matplotlib.pyplot as plt
import os

def train_model(optimizer_type='adam', epochs=100, lr=0.01, save_plots=True):
    """
    Train the TabGNN model with specified optimizer
    
    Parameters:
    - optimizer_type: 'adam' or 'eo' (Equilibrium Optimizer)
    - epochs: number of training epochs
    - lr: learning rate
    - save_plots: whether to save training plots
    
    Returns:
    - trained model
    - training history
    """
    # Load and preprocess data
    X, y, _ = load_and_preprocess('data/supply_chain_emission_factors.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build graph for training
    train_data = build_graph(X_train, y_train)
    test_data = build_graph(X_test, y_test)
    
    # Initialize model
    input_dim = X.shape[1]
    model = TabGNN(input_dim=input_dim)
    
    # Initialize optimizer based on type
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'eo':
        # Initialize Equilibrium Optimizer
        optimizer = EquilibriumOptimizer(
            model.parameters(),
            lr=lr,
            alpha=2.0,
            beta=0.1,
            c1=1.0,
            c2=1.0,
            c3=2.0,
            max_iter=epochs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Use 'adam' or 'eo'.")
    
    # Loss function
    loss_fn = torch.nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Training loop
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training step
        optimizer.zero_grad()
        out = model(train_data)
        loss = loss_fn(out, train_data.y)
        loss.backward()
        optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_out = model(test_data)
            val_loss = loss_fn(val_out, test_data.y)
        model.train()
        
        # Record history
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'results/best_model.pth')
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
    
    # Save training plots if requested
    if save_plots:
        save_training_plots(history, optimizer_type)
    
    return model, history

def save_training_plots(history, optimizer_type):
    """Save training history plots"""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Training History with {optimizer_type.upper()} Optimizer')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/training_history_{optimizer_type}.png')
    plt.close()

if __name__ == "__main__":
    # Train with Adam optimizer (default)
    print("Training with Adam optimizer...")
    train_model(optimizer_type='adam')
    
    # Train with Equilibrium Optimizer
    print("\nTraining with Equilibrium Optimizer...")
    train_model(optimizer_type='eo')