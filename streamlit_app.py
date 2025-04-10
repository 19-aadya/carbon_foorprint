# streamlit_app.py
import streamlit as st
import pandas as pd
import torch
import numpy as np
import os
from models.tabgnn_model import TabGNN
from utils.preprocessing import load_and_preprocess
from utils.graph_utils import build_graph
from models.train_pipeline import train_model
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Carbon Footprint Prediction",
    page_icon="🌱",
    layout="wide"
)

# App Title
st.title("🌱 Carbon Footprint Prediction using TabGNN")
st.markdown("Predicting Emission Factors for Different Industry Sectors with Graph Neural Networks")

# Sidebar for model options
st.sidebar.header("Model Options")
optimizer_choice = st.sidebar.selectbox(
    "Choose Optimizer", 
    ["Adam (Default)", "Equilibrium Optimizer (EO)"],
    index=0
)

# Map selection to internal names
optimizer_map = {
    "Adam (Default)": "adam",
    "Equilibrium Optimizer (EO)": "eo"
}

# Load and preprocess data
X, y, df = load_and_preprocess("data/supply_chain_emission_factors.csv")

# Build the graph data for GNN
data = build_graph(X, y)

# Load trained model with correct input dimension
input_dim = data.x.shape[1]

# Create models directory
os.makedirs("results", exist_ok=True)

# Check if models exist, otherwise train them
model_files = {
    "adam": "results/best_model_adam.pth",
    "eo": "results/best_model_eo.pth"
}

# Train tab
if st.sidebar.checkbox("Train New Models"):
    st.sidebar.warning("Training new models will take some time.")
    
    train_epochs = st.sidebar.slider("Training Epochs", min_value=10, max_value=500, value=100)
    train_lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.01, format="%.4f")
    
    if st.sidebar.button("Train Models"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train Adam model
        status_text.text("Training with Adam optimizer...")
        adam_model, adam_history = train_model(
            optimizer_type='adam', 
            epochs=train_epochs, 
            lr=train_lr
        )
        torch.save(adam_model.state_dict(), model_files["adam"])
        progress_bar.progress(50)
        
        # Train EO model
        status_text.text("Training with Equilibrium Optimizer...")
        eo_model, eo_history = train_model(
            optimizer_type='eo', 
            epochs=train_epochs, 
            lr=train_lr
        )
        torch.save(eo_model.state_dict(), model_files["eo"])
        progress_bar.progress(100)
        
        status_text.text("Training complete!")
        
        # Display training plots
        if os.path.exists("results/training_history_adam.png") and os.path.exists("results/training_history_eo.png"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Adam Training History")
                st.image("results/training_history_adam.png")
            with col2:
                st.subheader("EO Training History")
                st.image("results/training_history_eo.png")

# Use the selected optimizer
selected_optimizer = optimizer_map[optimizer_choice]
model_path = model_files[selected_optimizer]

# Initialize model
model = TabGNN(input_dim=input_dim)

# Try to load the model
try:
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        # If model doesn't exist, use default model path
        model.load_state_dict(torch.load("results/best_model.pth"))
    model.eval()
except RuntimeError as e:
    st.error(f"Error loading model: {str(e)}")
    st.error("Please train the models first using the sidebar options.")
    st.stop()

# Perform prediction on the full graph
with torch.no_grad():
    predictions = model(data).numpy()

# Add predictions to dataframe
df["Predicted Emission Factor"] = predictions

# Main content tabs
tab1, tab2, tab3 = st.tabs(["Predictions", "Visualization", "Calculator"])

# Tab 1: Predictions
with tab1:
    st.subheader("📊 Predictions on Sample Data")
    st.dataframe(df[["Sector", "EmissionFactor", "Predicted Emission Factor"]].head(10))
    
    # Dropdown for custom prediction
    st.subheader("🔍 Predict Emission for a Specific Sector")
    selected_sector = st.selectbox("Select a sector", df["Sector"].unique())
    
    if st.button("Predict Emission Factor"):
        selected_index = df[df["Sector"] == selected_sector].index[0]
        predicted_value = predictions[selected_index]
        actual_value = df.loc[selected_index, "EmissionFactor"]
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
    
    ax.bar(x - width/2, sector_data["EmissionFactor"], width, label="Actual", color="blue")
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
    
    sns.scatterplot(x=df["EmissionFactor"], y=df["Predicted Emission Factor"], ax=ax2)
    
    # Add diagonal line (perfect predictions)
    min_val = min(df["EmissionFactor"].min(), df["Predicted Emission Factor"].min())
    max_val = max(df["EmissionFactor"].max(), df["Predicted Emission Factor"].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")
    
    ax2.set_xlabel("Actual Emission Factor")
    ax2.set_ylabel("Predicted Emission Factor")
    ax2.set_title(f"Actual vs Predicted Values (Using {optimizer_choice})")
    ax2.legend()
    
    st.pyplot(fig2)
    
    # Error distribution
    st.subheader("📉 Prediction Error Distribution")
    
    df["Error"] = df["EmissionFactor"] - df["Predicted Emission Factor"]
    df["Absolute Error"] = df["Error"].abs()
    df["Percentage Error"] = (df["Absolute Error"] / df["EmissionFactor"]) * 100
    
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

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This app uses Graph Neural Networks to predict carbon emission factors for different industry sectors.
    
    The TabGNN model can be trained with either the standard Adam optimizer or the 
    Equilibrium Optimizer (EO), a physics-based metaheuristic algorithm.
    """
)