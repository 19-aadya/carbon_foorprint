import os
import pandas as pd

data_path = "data/Chapter Tables"

# Function to clean each file
def clean_csv(file_path):
    print(f"\n🔍 Processing {file_path}...\n")

    # Read first 15 rows to analyze structure
    preview_df = pd.read_csv(file_path, encoding="ISO-8859-1", header=None, nrows=15)

    # Find first non-empty row (potential header row)
    header_row = preview_df.dropna(how='all').index[0]
    print(f"🟢 Using row {header_row} as header\n")

    # Load the actual data using identified header
    df = pd.read_csv(file_path, encoding="ISO-8859-1", skiprows=header_row)

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # Rename columns
    df.columns = [f"col_{i}" for i in range(len(df.columns))]

    # Convert numeric columns where possible
    df = df.apply(pd.to_numeric, errors='ignore')

    print(df.head())  # Show cleaned data preview
    return df

# Process all CSV files
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            clean_csv(file_path)
