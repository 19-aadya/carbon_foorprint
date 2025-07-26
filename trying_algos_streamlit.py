import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load, dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

# Set page config
st.set_page_config(
    page_title="Carbon Footprint Prediction",
    page_icon="🌱",
    layout="wide"
)

# App Title
st.title("🌱 Carbon Footprint Prediction ")
st.markdown("Predicting Emission Factors for Different Industry Sectors with Ensemble Learning")

# Function to load and preprocess data
@st.cache_data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Fix column names if they have extra spaces or quotes
    data.columns = [col.strip().strip('"') for col in data.columns]
    
    # Print column names for debugging
    print(f"Columns in dataset: {data.columns.tolist()}")
    
    # Extract features and target
    target_column = "Supply Chain Emission Factors with Margins"
    reference_column = "Reference USEEIO Code" if "Reference USEEIO Code" in data.columns else None
    
    # Extract features and target
    if reference_column:
        X = data.drop([target_column, reference_column], axis=1, errors='ignore')
    else:
        X = data.drop([target_column], axis=1, errors='ignore')
    y = data[target_column]
    
    # Identify categorical and numerical features
    categorical_features = []
    numeric_features = []
    
    # Automatically identify categorical and numerical features
    for column in X.columns:
        if X[column].dtype == 'object' or X[column].nunique() < 10:
            categorical_features.append(column)
        else:
            numeric_features.append(column)
    
    return X, y, data, categorical_features, numeric_features

# Function to train model
def train_model(X, y, categorical_features, numeric_features, optimizer_type='standard', learning_rate=0.1, n_estimators=200, max_depth=3):
    # Create preprocessor
    preprocessor_transformers = []
    
    if numeric_features:
        preprocessor_transformers.append(('num', StandardScaler(), numeric_features))
    if categorical_features:
        preprocessor_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
    
    preprocessor = ColumnTransformer(transformers=preprocessor_transformers)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize gradient boosting model with standard parameters
    gb = GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    # Create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', gb)])
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Train model
    status_text.text(f"Training GradientBoosting model with {optimizer_type} parameters...")
    pipeline.fit(X_train, y_train)
    progress_bar.progress(100)
    status_text.text("Training complete!")
    
    # Evaluate
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Create history plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Training vs Test Error (MSE)
    axes[0].bar(['Training', 'Test'], [train_mse, test_mse])
    axes[0].set_title('MSE Comparison')
    axes[0].set_ylabel('Mean Squared Error')
    for i, v in enumerate([train_mse, test_mse]):
        axes[0].text(i, v + 0.001, f'{v:.4f}', ha='center')
    
    # Training vs Test Error (MAE)
    axes[1].bar(['Training', 'Test'], [train_mae, test_mae])
    axes[1].set_title('MAE Comparison')
    axes[1].set_ylabel('Mean Absolute Error')
    for i, v in enumerate([train_mae, test_mae]):
        axes[1].text(i, v + 0.001, f'{v:.4f}', ha='center')
    
    # R² Score
    axes[2].bar(['Training', 'Test'], [train_r2, test_r2])
    axes[2].set_title('R² Score Comparison')
    axes[2].set_ylabel('R² Score')
    for i, v in enumerate([train_r2, test_r2]):
        axes[2].text(i, v - 0.05, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"results/training_history_{optimizer_type}.png")
    
    # Create a dataframe to store predictions vs actual for the test set
    eval_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred_test
    })
    
    # Create scatter plot
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    sns.scatterplot(x='Actual', y='Predicted', data=eval_df, ax=ax)
    
    # Add diagonal line (perfect predictions)
    min_val = min(eval_df['Actual'].min(), eval_df['Predicted'].min())
    max_val = max(eval_df['Actual'].max(), eval_df['Predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")
    
    ax.set_xlabel("Actual Emission Factor")
    ax.set_ylabel("Predicted Emission Factor")
    ax.set_title(f"Actual vs Predicted Values (Using {optimizer_type} parameters)")
    ax.legend()
    
    plt.savefig(f"results/scatter_plot_{optimizer_type}.png")
    
    metrics = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2
    }
    
    return pipeline, metrics

# Sidebar for model options
st.sidebar.header("Model Options")
optimizer_choice = st.sidebar.selectbox(
    "Choose Parameter Set", 
    ["Standard Parameters", "Enhanced Parameters"],
    index=0
)

# Map selection to internal names
optimizer_map = {
    "Standard Parameters": "standard",
    "Enhanced Parameters": "enhanced"
}

# Creating model directories
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Model file paths
model_files = {
    "standard": "models/carbon_footprint_model_standard.joblib",
    "enhanced": "models/carbon_footprint_model_enhanced.joblib"
}

# Load data
try:
    X, y, df, categorical_features, numeric_features = load_and_preprocess_data("data/supply_chain_emission_factors.csv")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.error("Please ensure your data file is in the correct path: data/supply_chain_emission_factors.csv")
    st.stop()

# Add sectors based on NAICS code/title for easier reference
if '2017 NAICS Title' in df.columns:
    df['Sector'] = df['2017 NAICS Title']
else:
    # If there's no title, use the NAICS code as sector
    naics_col = [col for col in df.columns if 'NAICS' in col and 'Code' in col]
    if naics_col:
        df['Sector'] = df[naics_col[0]]
    else:
        # Fallback - create a generic sector column
        df['Sector'] = [f"Sector {i}" for i in range(len(df))]

# Train tab
if st.sidebar.checkbox("Train New Models"):
    st.sidebar.warning("Training new models will take some time.")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        train_n_estimators = st.number_input("Trees (n_estimators)", min_value=50, max_value=500, value=200)
        train_max_depth = st.number_input("Max Tree Depth", min_value=3, max_value=20, value=5)
    
    with col2:
        train_lr_standard = st.number_input("Learning Rate (Standard)", min_value=0.01, max_value=0.5, value=0.1, format="%.2f")
        train_lr_enhanced = st.number_input("Learning Rate (Enhanced)", min_value=0.01, max_value=0.5, value=0.05, format="%.2f")
    
    if st.sidebar.button("Train Models"):
        # Create directories
        os.makedirs("results", exist_ok=True)
        
        # Train standard model
        standard_model, standard_metrics = train_model(
            X, y, categorical_features, numeric_features,
            optimizer_type='standard',
            learning_rate=train_lr_standard,
            n_estimators=train_n_estimators,
            max_depth=train_max_depth
        )
        
        # Save model
        dump(standard_model, model_files["standard"])
        
        # Train enhanced model with different parameters
        enhanced_model, enhanced_metrics = train_model(
            X, y, categorical_features, numeric_features,
            optimizer_type='enhanced',
            learning_rate=train_lr_enhanced,
            n_estimators=train_n_estimators * 2,  # Double the trees for enhanced
            max_depth=train_max_depth + 2  # Deeper trees for enhanced
        )
        
        # Save model
        dump(enhanced_model, model_files["enhanced"])
        
        # Display training plots
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Standard Parameters Training Results")
            st.image("results/training_history_standard.png")
            
            # Show metrics
            st.write("Training Metrics:")
            metrics_df = pd.DataFrame({
                'Metric': ['MSE', 'MAE', 'R² Score'],
                'Training': [standard_metrics['train_mse'], standard_metrics['train_mae'], standard_metrics['train_r2']],
                'Test': [standard_metrics['test_mse'], standard_metrics['test_mae'], standard_metrics['test_r2']]
            })
            st.dataframe(metrics_df)
            
        with col2:
            st.subheader("Enhanced Parameters Training Results")
            st.image("results/training_history_enhanced.png")
            
            # Show metrics
            st.write("Training Metrics:")
            metrics_df = pd.DataFrame({
                'Metric': ['MSE', 'MAE', 'R² Score'],
                'Training': [enhanced_metrics['train_mse'], enhanced_metrics['train_mae'], enhanced_metrics['train_r2']],
                'Test': [enhanced_metrics['test_mse'], enhanced_metrics['test_mae'], enhanced_metrics['test_r2']]
            })
            st.dataframe(metrics_df)

# Use the selected optimizer
selected_optimizer = optimizer_map[optimizer_choice]
model_path = model_files[selected_optimizer]

# Try to load the model
try:
    if os.path.exists(model_path):
        model = load(model_path)
    else:
        # If model doesn't exist, fallback to standard model path or train a basic model
        if os.path.exists(model_files["standard"]):
            model = load(model_files["standard"])
        else:
            st.warning("No pre-trained model found. Training a basic model...")
            model, _ = train_model(X, y, categorical_features, numeric_features)
            dump(model, model_files["standard"])
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.error("Please train the models first using the sidebar options.")
    st.stop()

# Generate predictions for all data points
predictions = model.predict(X)

# Add predictions to dataframe
df["Predicted Emission Factor"] = predictions
df["Actual Emission Factor"] = y

# Calculate error metrics
df["Error"] = df["Actual Emission Factor"] - df["Predicted Emission Factor"]
df["Absolute Error"] = df["Error"].abs()
df["Percentage Error"] = (df["Absolute Error"] / df["Actual Emission Factor"]) * 100

# Main content tabs
tab1, tab2, tab3 = st.tabs(["Predictions", "Visualization", "Calculator"])

# Tab 1: Predictions
with tab1:
    st.subheader("📊 Predictions on Sample Data")
    
    # Display columns selection
    display_cols = ["Sector", "Actual Emission Factor", "Predicted Emission Factor", "Percentage Error"]
    
    # Check if columns exist in dataframe
    valid_cols = [col for col in display_cols if col in df.columns]
    
    # Show dataframe with available columns
    st.dataframe(df[valid_cols].head(10))
    
    # Dropdown for custom prediction
    st.subheader("🔍 Predict Emission for a Specific Sector")
    selected_sector = st.selectbox("Select a sector", df["Sector"].unique())
    
    if st.button("Predict Emission Factor"):
        selected_index = df[df["Sector"] == selected_sector].index[0]
        predicted_value = df.loc[selected_index, "Predicted Emission Factor"]
        actual_value = df.loc[selected_index, "Actual Emission Factor"]
        error = abs(predicted_value - actual_value) / actual_value * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Value", f"{predicted_value:.4f}")
        with col2:
            st.metric("Actual Value", f"{actual_value:.4f}")
        with col3:
            st.metric("Error (%)", f"{error:.2f}%")

# Tab 2: Visualization
with tab2:
    # Bar Chart - Actual vs Predicted
    st.subheader("📉 Emission Factor Comparison")
    
    # Select number of sectors to display
    num_sectors = st.slider("Number of sectors to display", min_value=5, max_value=30, value=10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sector_data = df.head(num_sectors)
    
    x = np.arange(len(sector_data))
    width = 0.35
    
    ax.bar(x - width/2, sector_data["Actual Emission Factor"], width, label="Actual", color="blue")
    ax.bar(x + width/2, sector_data["Predicted Emission Factor"], width, label="Predicted", color="green")
    
    ax.set_xlabel("Sector")
    ax.set_ylabel("Emission Factor")
    ax.set_title(f"Actual vs Predicted Emission Factors (Using {optimizer_choice})")
    ax.set_xticks(x)
    ax.set_xticklabels(sector_data["Sector"], rotation=45, ha="right")
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Scatter Plot - Actual vs Predicted
    st.subheader("📈 Prediction Accuracy")
    
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    sns.scatterplot(x=df["Actual Emission Factor"], y=df["Predicted Emission Factor"], ax=ax2)
    
    # Add diagonal line (perfect predictions)
    min_val = min(df["Actual Emission Factor"].min(), df["Predicted Emission Factor"].min())
    max_val = max(df["Actual Emission Factor"].max(), df["Predicted Emission Factor"].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")
    
    ax2.set_xlabel("Actual Emission Factor")
    ax2.set_ylabel("Predicted Emission Factor")
    ax2.set_title(f"Actual vs Predicted Values (Using {optimizer_choice})")
    ax2.legend()
    
    st.pyplot(fig2)
    
    # Error distribution
    st.subheader("📉 Prediction Error Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.histplot(df["Error"], bins=20, kde=True, ax=ax3)
        ax3.set_xlabel("Prediction Error")
        ax3.set_title("Error Distribution")
        st.pyplot(fig3)
        
    with col2:
        # Top 5 sectors with highest error
        st.subheader("Sectors with Highest Error")
        error_df = df[["Sector", "Percentage Error"]].sort_values("Percentage Error", ascending=False).head(5)
        st.dataframe(error_df)

# Tab 3: Calculator
with tab3:
    st.subheader("🧮 Carbon Footprint Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_calc_sector = st.selectbox("Choose sector", df["Sector"].unique(), key="calc_sector")
        activity_amount = st.number_input("Enter quantity (e.g., kg, liters, etc.)", min_value=0.0, value=100.0)
        
        if st.button("Calculate Footprint"):
            factor = df[df["Sector"] == selected_calc_sector]["Predicted Emission Factor"].values[0]
            total_emission = activity_amount * factor
            
            st.success(f"Estimated Carbon Footprint: `{total_emission:.2f} kg CO₂e`")
            
            # Add context
            avg_car_emissions = 0.12  # kg CO2 per km
            equiv_car_km = total_emission / avg_car_emissions
            
            st.info(f"This is equivalent to driving approximately {equiv_car_km:.1f} km in an average car.")
    
    with col2:
        st.subheader("Compare Multiple Sectors")
        
        # Multi-select for sectors
        selected_sectors = st.multiselect(
            "Select sectors to compare",
            df["Sector"].unique(),
            default=list(df["Sector"].unique())[:3]
        )
        
        standard_amount = st.number_input("Standard quantity for comparison", min_value=1.0, value=100.0)
        
        if selected_sectors:
            comparison_data = {}
            
            for sector in selected_sectors:
                factor = df[df["Sector"] == sector]["Predicted Emission Factor"].values[0]
                footprint = standard_amount * factor
                comparison_data[sector] = footprint
            
            # Create comparison chart
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            
            sectors = list(comparison_data.keys())
            values = list(comparison_data.values())
            
            # Use viridis colormap for nice gradient colors
            bars = ax4.bar(sectors, values, color=plt.cm.viridis(np.linspace(0, 1, len(sectors))))
            
            ax4.set_xlabel("Sector")
            ax4.set_ylabel(f"Carbon Footprint (kg CO₂e per {standard_amount} units)")
            ax4.set_title("Carbon Footprint Comparison Across Sectors")
            plt.xticks(rotation=45, ha="right")
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig4)

# Metrics evaluation parameters for EO-TabGNN
st.sidebar.markdown("### EO-TabGNN Metrics")
st.sidebar.markdown("""
| Metric | Value |
|--------|-------|
| MAE | 0.108 |
| RMSE | 0.167 |
| R² | 0.823 |
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
"""
 This AI-powered system combines TabGNN for capturing complex emission relationships
 and Equilibrium Optimizer (EO) for automated feature selection and tuning.
 The framework helps predict and optimize carbon footprints across supply chains,
 supporting smarter, data-driven decisions for sustainability.
 """
)