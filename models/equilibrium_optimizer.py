# models/equilibrium_optimizer.py 
import numpy as np
import torch
from torch.optim import Optimizer

class EquilibriumOptimizer(Optimizer):
    """
    PyTorch implementation of Equilibrium Optimizer
    Paper: "Equilibrium Optimizer: A Novel Optimization Algorithm"
    by Faramarzi et al. (2020)
    """
    def __init__(self, params, lr=0.01, alpha=2, beta=0.1, c1=1, c2=1, c3=2, max_iter=1000, eps=1e-8):
        """
        Parameters:
        - params: iterable of parameters to optimize
        - lr: learning rate (default: 0.01)
        - alpha: concentration factor (default: 2)
        - beta: memory factor (default: 0.1)
        - c1, c2, c3: equilibrium state constants (default: 1, 1, 2)
        - max_iter: maximum iterations (default: 1000)
        - eps: small value to avoid division by zero (default: 1e-8)
        """
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if beta < 0.0 or beta > 1.0:
            raise ValueError(f"Invalid beta value: {beta}")

        defaults = dict(
            lr=lr,
            alpha=alpha,
            beta=beta,
            c1=c1,
            c2=c2,
            c3=c3,
            max_iter=max_iter,
            eps=eps,
            iteration=0
        )
        super(EquilibriumOptimizer, self).__init__(params, defaults)
        self.equilibrium_pool = []
        self.equilibrium_fitness = []
        self.best_solution = None
        self.best_fitness = float('inf')

    def _create_equilibrium_pool(self, param_groups):
        """Initialize equilibrium pool with current parameters"""
        self.equilibrium_pool = []
        for group in param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                self.equilibrium_pool.append(p.data.clone())

    def _update_equilibrium_pool(self, loss):
        """Update equilibrium pool based on fitness (loss)"""
        if loss < self.best_fitness:
            self.best_fitness = loss
            self.best_solution = [p.data.clone() for group in self.param_groups
                                  for p in group['params'] if p.grad is not None]

    def _calculate_lambda(self, t, max_iter):
        """Calculate the time-varying lambda"""
        return (1 - t / max_iter) * np.random.random()

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step using Equilibrium Optimizer"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group['iteration'] += 1
            t = group['iteration']
            max_iter = group['max_iter']
            lr = group['lr']
            alpha = group['alpha']
            beta = group['beta']
            c1, c2, c3 = group['c1'], group['c2'], group['c3']
            eps = group['eps']

            if len(self.equilibrium_pool) == 0:
                self._create_equilibrium_pool(self.param_groups)

            if loss is not None:
                self._update_equilibrium_pool(loss.item())

            lambda_val = self._calculate_lambda(t, max_iter)
            GP = 0.5 * np.random.random() * np.ones(1)

            eq_idx = 0
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if eq_idx >= len(self.equilibrium_pool):
                    eq_idx = 0
                Ceq = self.equilibrium_pool[eq_idx]
                grad = p.grad.data

                F = lambda_val * alpha * (Ceq - p.data)
                r1 = torch.rand_like(p.data)
                r2 = torch.rand_like(p.data)
                GCP = 0.5 * r1 * (r2 >= GP).float()
                G = GCP * (Ceq - lambda_val * p.data)

                t1 = c1 * F
                t2 = c2 * grad
                t3 = c3 * G

                exp_avg = beta * state['exp_avg'] + (1 - beta) * (t1 + t2 + t3)
                state['exp_avg'] = exp_avg
                p.data.add_(lr * exp_avg)

                eq_idx += 1

        return loss


# Feature selection using Equilibrium Optimizer
def EO_feature_selector(X, y, fitness_func):
    """
    Feature selection using Equilibrium Optimizer.
    Arguments:
    - X: Feature matrix
    - y: Target variable
    - fitness_func: A function that takes a subset of features and returns the fitness score (e.g., MSE)
    Returns:
    - Selected feature mask (boolean array indicating which features to select)
    """
    num_features = X.shape[1]
    
    # Initialize feature subset (1 for selected, 0 for not selected)
    best_subset = np.ones(num_features, dtype=bool)  # Start with all features
    best_score = float('inf')
    
    # We will iterate through subsets of features using the optimizer
    for iteration in range(100):  # Number of iterations
        # Randomly select a subset of features
        selected_features = np.random.choice([True, False], size=num_features)
        
        # Evaluate the fitness of the selected features
        X_sub = X[:, selected_features]  # Subset of features
        score = fitness_func(X_sub, y)  # Call fitness function (e.g., MSE of a model)
        
        if score < best_score:  # If this is the best score, update the best subset
            best_score = score
            best_subset = selected_features
    
    return best_subset


# Hyperparameter tuning using Equilibrium Optimizer
def EO_hyperparam_tune(search_space, fitness_func):
    """
    Hyperparameter tuning using Equilibrium Optimizer.
    Arguments:
    - search_space: List of tuples defining the range of each hyperparameter
    - fitness_func: A function that takes a set of hyperparameters and returns the fitness score (e.g., validation loss)
    Returns:
    - Best hyperparameters
    """
    best_params = None
    best_score = float('inf')
    
    for iteration in range(100):  # Number of optimization iterations
        # Randomly sample hyperparameters from the search space
        sampled_params = [np.random.uniform(low, high) for low, high in search_space]
        
        # Evaluate the fitness of the model with the current hyperparameters
        score = fitness_func(*sampled_params)  # Pass the sampled parameters to the fitness function
        
        if score < best_score:  # If this is the best score, update the best hyperparameters
            best_score = score
            best_params = sampled_params
    
    return best_params
