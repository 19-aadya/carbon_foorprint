import pandas as pd
import joblib

# Load model
model = joblib.load("models/lgbm_regressor.pkl")

# Load new data for prediction
df = pd.read_csv("data/processed_data.csv")
X = df.drop(columns=['CarbonEmission'], errors='ignore')  # Drop target variable if it exists

# Predict with the model
predictions = model.predict(X)

# Save predictions
df['Predicted_Carbon_Emission'] = predictions
df.to_csv("data/predictions.csv", index=False)
print("Predictions saved to predictions.csv")
