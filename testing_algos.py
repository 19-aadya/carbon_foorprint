import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data/supply_chain_emission_factors.csv')

# Display basic information about the dataset
print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())
print("\nData types:")
print(data.dtypes)
print("\nSummary statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Fix column names if they have extra spaces or quotes
data.columns = [col.strip().strip('"') for col in data.columns]

# Extract features and target
# Make sure we use the exact column names as in the dataset
target_column = "Supply Chain Emission Factors with Margins"
reference_column = "Reference USEEIO Code" if "Reference USEEIO Code" in data.columns else None

# List columns to check what we actually have
print("\nAvailable columns:")
for col in data.columns:
    print(f"- {col}")

# Identify target column by searching for partial match if exact match not found
if target_column not in data.columns:
    potential_targets = [col for col in data.columns if "with Margins" in col]
    if potential_targets:
        target_column = potential_targets[0]
        print(f"\nUsing '{target_column}' as target column")

# Extract features and target
if reference_column:
    X = data.drop([target_column, reference_column], axis=1, errors='ignore')
else:
    X = data.drop([target_column], axis=1, errors='ignore')
y = data[target_column]

# Identify categorical and numerical columns
# First, let's check the data types
print("\nFeature data types:")
print(X.dtypes)

# Define categorical and numerical features
categorical_features = []
numeric_features = []

# Automatically identify categorical and numerical features
for column in X.columns:
    if X[column].dtype == 'object' or X[column].nunique() < 10:
        categorical_features.append(column)
    else:
        numeric_features.append(column)

print("\nIdentified categorical features:", categorical_features)
print("Identified numerical features:", numeric_features)

# Try to convert NAICS code to numeric if it exists and is not already numeric
naics_column = None
for col in X.columns:
    if 'NAICS' in col and 'Code' in col:
        naics_column = col
        break

if naics_column:
    # Check if it's already numeric
    if X[naics_column].dtype != 'object':
        print(f"\n{naics_column} is already numeric with dtype: {X[naics_column].dtype}")
    else:
        # Try to extract numeric part
        try:
            X[naics_column + '_Numeric'] = pd.to_numeric(
                X[naics_column].astype(str).str.extract('(\d+)', expand=False), 
                errors='coerce'
            )
            numeric_features.append(naics_column + '_Numeric')
            print(f"Created numeric version of {naics_column}")
        except Exception as e:
            print(f"Could not convert {naics_column} to numeric: {e}")

# Create preprocessor
preprocessor_transformers = []

if numeric_features:
    preprocessor_transformers.append(('num', StandardScaler(), numeric_features))
if categorical_features:
    preprocessor_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))

preprocessor = ColumnTransformer(transformers=preprocessor_transformers)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to try
models = {
    'ElasticNet': ElasticNet(random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42)
}

# Function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

# Dictionary to store results
results = {}

# Train and evaluate models
for name, model in models.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    eval_results = evaluate_model(pipeline, X_test, y_test)
    results[name] = eval_results
    
    print(f"{name} evaluation:")
    for metric, value in eval_results.items():
        print(f"  {metric}: {value:.4f}")

# Find the best model based on R2 score
best_model_name = max(results, key=lambda x: results[x]['R2'])
print(f"\nBest model: {best_model_name}")
print(f"R2 Score: {results[best_model_name]['R2']:.4f}")
print(f"MSE: {results[best_model_name]['MSE']:.4f}")
print(f"MAE: {results[best_model_name]['MAE']:.4f}")

# Hyperparameter tuning for the best model
print("\nPerforming hyperparameter tuning for", best_model_name)

if best_model_name == 'ElasticNet':
    param_grid = {
        'model__alpha': [0.001, 0.01, 0.1, 1, 10],
        'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
elif best_model_name == 'RandomForest':
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'GradientBoosting':
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    }
elif best_model_name == 'XGBoost':
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__colsample_bytree': [0.7, 0.8, 0.9]
    }

# Create the best model pipeline
best_model = models[best_model_name]
best_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])

# Grid search with cross-validation
grid_search = GridSearchCV(best_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
best_tuned_model = grid_search.best_estimator_

# Evaluate the tuned model
tuned_results = evaluate_model(best_tuned_model, X_test, y_test)
print("\nTuned model evaluation:")
for metric, value in tuned_results.items():
    print(f"  {metric}: {value:.4f}")

# Plot actual vs predicted values
y_pred = best_tuned_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.savefig('actual_vs_predicted.png')

# Create a function for carbon footprint calculation
def calculate_carbon_footprint(activity_data, model=best_tuned_model):
    """
    Calculate carbon footprint based on activity data
    
    Parameters:
    activity_data (DataFrame): DataFrame containing activity information
    model: Trained model to predict emission factors
    
    Returns:
    DataFrame: Carbon footprint calculations
    """
    # Make predictions using the model
    emission_factors = model.predict(activity_data)
    
    # Create result DataFrame with basic information
    result = pd.DataFrame({
        'Predicted_Emission_Factor': emission_factors
    })
    
    # Add activity information if available
    for col in activity_data.columns:
        if col in categorical_features:
            result[col] = activity_data[col]
    
    # If activity quantity is provided, calculate the carbon footprint
    if 'Activity_Quantity' in activity_data.columns:
        result['Carbon_Footprint'] = activity_data['Activity_Quantity'] * emission_factors
    
    return result

# Save the best model
from joblib import dump
dump(best_tuned_model, 'carbon_footprint_model.joblib')

print("\nModel saved as 'carbon_footprint_model.joblib'")
print("You can now use this model to calculate carbon footprints based on activity data.")

# Example usage
print("\nExample: To calculate carbon footprint, you would do:")
print("from joblib import load")
print("model = load('carbon_footprint_model.joblib')")
print("activity_data = pd.DataFrame({...})  # Your activity data")
print("footprint = calculate_carbon_footprint(activity_data, model)")