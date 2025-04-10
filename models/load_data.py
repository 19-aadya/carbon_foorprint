import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_dataset(file_name="supply_chain_emission_factors.csv", target_column="Supply Chain Emission Factors with Margins"):
    """Loads and prepares the dataset for training and evaluation."""

    # Path to the data file
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", file_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # Load CSV
    df = pd.read_csv(data_path)

    # Check target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset!")

    # Select only numeric features
    feature_cols = [
        "Supply Chain Emission Factors without Margins",
        "Margins of Supply Chain Emission Factors"
    ]

    # Features and Target
    X = df[feature_cols]
    y = df[target_column]

    return train_test_split(X, y, test_size=0.2, random_state=42)
