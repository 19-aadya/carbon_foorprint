import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load preprocessed dataset
df = pd.read_csv("data/processed_data.csv")

# Define features and target variable
X = df.drop(columns=["CarbonEmission"])  # Adjust based on actual target column name
y = df["CarbonEmission"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM model
model = LGBMRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Convert errors to percentage
mae_pct = (mae / y_test.mean()) * 100
mse_pct = (mse / (y_test.mean() ** 2)) * 100
rmse_pct = (rmse / y_test.mean()) * 100

# Save trained model
joblib.dump(model, "models/lgbm_model.pkl")

# Print performance metrics
print(f"Model training completed and saved.")
print(f"Mean Absolute Error (MAE): {mae:.2f} ({mae_pct:.2f}%)")
print(f"Mean Squared Error (MSE): {mse:.2f} ({mse_pct:.2f}%)")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} ({rmse_pct:.2f}%)")
print(f"R² Score: {r2:.2f}")
