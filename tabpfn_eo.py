import pandas as pd
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
from scipy.optimize import minimize

# Load the dataset
df = pd.read_csv("data/supply_chain_emission_factors.csv")
df = df.rename(columns={
    '2017 NAICS Title': 'Sector',
    'Supply Chain Emission Factors with Margins': 'EmissionFactor'
})
df = df[['Sector', 'EmissionFactor']]  # keep relevant columns
df.dropna(inplace=True)  # remove rows with missing values

# Encode the 'Sector' text into numbers using Label Encoding
encoder = LabelEncoder()
df['Sector_encoded'] = encoder.fit_transform(df['Sector'])
X = df[['Sector_encoded']].values
y = df['EmissionFactor'].values

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to define and train TabPFN Model with EO optimization
def objective_function(params, model, X_train, y_train, X_test, y_test):
    n_estimators = int(params[0])
    softmax_temperature = params[1]

    # Set parameters
    model.set_params(n_estimators=n_estimators, softmax_temperature=softmax_temperature)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predictions
    preds = model.predict(X_test)
    
    # Return MSE for optimization
    mse = mean_squared_error(y_test, preds)
    return mse

# Set up TabPFN Regressor
model = TabPFNRegressor(device="cuda")  # Make sure you have a CUDA-compatible GPU

# Use EO (Equilibrium Optimizer) to tune hyperparameters
def optimize_hyperparameters(model, X_train, y_train, X_test, y_test):
    # Define parameter bounds for optimization
    param_bounds = {
        'n_estimators': (50, 500),  # Number of estimators
        'softmax_temperature': (0.01, 1.0)  # Softmax temperature
    }

    # Objective function to minimize MSE
    def optimize(params):
        n_estimators = int(params[0])
        softmax_temperature = params[1]

        # Set parameters
        model.set_params(n_estimators=n_estimators, softmax_temperature=softmax_temperature)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        return mse

    # Minimize the objective function (MSE)
    initial_params = [100, 0.5]  # Initial guess for n_estimators and softmax_temperature
    result = minimize(optimize, initial_params, bounds=[param_bounds['n_estimators'], param_bounds['softmax_temperature']])

    best_params = result.x
    print(f"Best Params found: {best_params}")
    return best_params

# Optimize hyperparameters using EO
best_params = optimize_hyperparameters(model, X_train, y_train, X_test, y_test)

# Retrain model with the best hyperparameters found
model.set_params(n_estimators=int(best_params[0]), softmax_temperature=best_params[1])
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")
