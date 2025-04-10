import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from equilibrium_optimizer import EquilibriumOptimizer  # Custom EO implementation
from hunger_games_search import HungerGamesSearch  # Custom HGS implementation

# Load preprocessed dataset
df = pd.read_csv("data/processed_data.csv")

# Define features and target variable
X = df.drop(columns=["CarbonEmission"])  # Adjust based on actual target column name
y = df["CarbonEmission"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Equilibrium Optimizer (EO) for Feature Selection
eo = EquilibriumOptimizer(num_agents=30, max_iterations=50, feature_size=X_train.shape[1])
selected_features = neo.optimize(X_train, y_train)
X_train_selected, X_test_selected = X_train.iloc[:, selected_features], X_test.iloc[:, selected_features]

# Apply Hunger Games Search (HGS) for Hyperparameter Optimization
hgs = HungerGamesSearch(param_grid={
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 10]
})

best_params = hgs.optimize(X_train_selected, y_train)

# Train optimized LightGBM model
from lightgbm import LGBMRegressor
model = LGBMRegressor(**best_params)
model.fit(X_train_selected, y_train)

# Make predictions
y_pred = model.predict(X_test_selected)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Save trained model
joblib.dump(model, "models/optimized_carbon_footprint_model.pkl")

# Print performance metrics
print("Model training completed and saved.")
print(f"Mean Absolute Error (MAE): {mae:.2%}")
print(f"Mean Squared Error (MSE): {mse:.2%}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2%}")
print(f"R² Score: {r2:.2%}")
