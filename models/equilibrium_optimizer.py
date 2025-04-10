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
        
        # Initialize equilibrium candidates
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
                if len(state) == 0:  # Initialize state
                    state['exp_avg'] = torch.zeros_like(p.data)
                
                # Store current param as candidate solution
                self.equilibrium_pool.append(p.data.clone())
                
    def _update_equilibrium_pool(self, loss):
        """Update equilibrium pool based on fitness (loss)"""
        if loss < self.best_fitness:
            self.best_fitness = loss
            self.best_solution = [p.data.clone() for group in self.param_groups 
                                 for p in group['params'] if p.grad is not None]
    
    def _calculate_lambda(self, t, max_iter):
        """Calculate the time-varying lambda"""
        return (1 - t/max_iter) * np.random.random()
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step using Equilibrium Optimizer"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Increment iteration counter
        for group in self.param_groups:
            group['iteration'] += 1
            t = group['iteration']
            max_iter = group['max_iter']
            lr = group['lr']
            alpha = group['alpha']
            beta = group['beta'] 
            c1, c2, c3 = group['c1'], group['c2'], group['c3']
            eps = group['eps']
            
            # Initialize equilibrium pool if needed
            if len(self.equilibrium_pool) == 0:
                self._create_equilibrium_pool(self.param_groups)
            
            # Update equilibrium pool based on current loss
            if loss is not None:
                self._update_equilibrium_pool(loss.item())
            
            # Calculate lambda (time-varying factor)
            lambda_val = self._calculate_lambda(t, max_iter)
            
            # Generation probability (GP)
            GP = 0.5 * np.random.random() * np.ones(1)
            
            eq_idx = 0  # Index to track equilibrium candidate
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if eq_idx >= len(self.equilibrium_pool):
                    eq_idx = 0
                    
                # Get equilibrium candidate
                Ceq = self.equilibrium_pool[eq_idx]
                
                # Get gradient
                grad = p.grad.data
                
                # Calculate F term
                F = lambda_val * alpha * (Ceq - p.data)
                
                # Calculate exponential term
                r1 = torch.rand_like(p.data)
                r2 = torch.rand_like(p.data)
                GCP = 0.5 * r1 * (r2 >= GP).float()
                
                # Calculate generation rate G
                G = GCP * (Ceq - lambda_val * p.data)
                
                # Calculate update term
                t1 = c1 * F
                t2 = c2 * grad
                t3 = c3 * G
                
                # Update parameters
                exp_avg = beta * state['exp_avg'] + (1 - beta) * (t1 + t2 + t3)
                state['exp_avg'] = exp_avg
                
                p.data.add_(lr * exp_avg)
                
                eq_idx += 1
                
        return loss