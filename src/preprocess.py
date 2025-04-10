import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    # Load dataset
    df = pd.read_csv("data/Carbon_Emission.csv")

    # Drop missing values (if any)
    df.dropna(inplace=True)

    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Save processed data
    df.to_csv("data/processed_data.csv", index=False)
    return df, label_encoders

if __name__ == "__main__":
    preprocess_data()
