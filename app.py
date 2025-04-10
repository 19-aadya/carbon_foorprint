import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = joblib.load("models/lgbm_model.pkl")

# Streamlit App
st.set_page_config(page_title="AI-Powered Carbon Footprint Estimator", layout="wide")
st.title("🌍 AI-Powered Carbon Footprint Estimator")
st.write("Enter your details to estimate your carbon emissions and visualize the insights.")

# User Input Form
st.sidebar.header("User Input")
body_type = st.sidebar.selectbox("🏋️ Body Type", ["Underweight", "Normal", "Overweight", "Obese"])
sex = st.sidebar.selectbox("🚻 Sex", ["Male", "Female"])
diet = st.sidebar.selectbox("🥗 Diet", ["Omnivore", "Vegetarian", "Vegan", "Pescatarian"])
how_often_shower = st.sidebar.selectbox("🚿 How Often Shower", ["Less frequently", "Daily", "More frequently", "Twice a day"])
heating_energy_source = st.sidebar.selectbox("🔥 Heating Energy Source", ["Coal", "Natural Gas", "Wood", "Electricity"])
transport = st.sidebar.selectbox("🚉 Transport", ["Private", "Public", "Walk/Bicycle"])
vehicle_type = st.sidebar.selectbox("🚗 Vehicle Type", ["Petrol", "Diesel", "Electric"])
social_activity = st.sidebar.selectbox("🎭 Social Activity", ["Never", "Sometimes", "Often"])
monthly_grocery_bill = st.sidebar.slider("🛒 Monthly Grocery Bill ($)", 0, 500, 100)
frequency_air_travel = st.sidebar.selectbox("✈️ Frequency of Traveling by Air", ["Never", "Rarely", "Sometimes", "Frequently"])
vehicle_monthly_distance = st.sidebar.slider("🚘 Vehicle Monthly Distance (Km)", 0, 5000, 1000)
waste_bag_size = st.sidebar.selectbox("🗑️ Waste Bag Size", ["Small", "Medium", "Large", "Extra Large"])
waste_bag_weekly_count = st.sidebar.slider("♻️ Waste Bag Weekly Count", 0, 10, 3)
how_long_tv_pc = st.sidebar.slider("📺 How Long TV/PC Daily (Hours)", 0, 24, 5)
how_many_new_clothes = st.sidebar.slider("👕 How Many New Clothes Monthly", 0, 50, 2)
how_long_internet = st.sidebar.slider("🌐 How Long Internet Daily (Hours)", 0, 24, 6)
energy_efficiency = st.sidebar.selectbox("💡 Energy Efficiency", ["No", "Sometimes", "Yes"])
recycling = st.sidebar.multiselect("♻️ Recycling", ["Paper", "Plastic", "Glass", "Metal"])
cooking_with = st.sidebar.multiselect("🍳 Cooking With", ["Stove", "Oven", "Microwave", "Grill", "Airfryer"])

# Encoding categorical variables
categorical_features = {
    "BodyType": body_type,
    "Sex": sex,
    "Diet": diet,
    "HowOftenShower": how_often_shower,
    "HeatingEnergySource": heating_energy_source,
    "Transport": transport,
    "VehicleType": vehicle_type,
    "SocialActivity": social_activity,
    "FrequencyAirTravel": frequency_air_travel,
    "WasteBagSize": waste_bag_size,
    "EnergyEfficiency": energy_efficiency,
    "Recycling": ",".join(recycling) if recycling else "None",
    "CookingWith": ",".join(cooking_with) if cooking_with else "None"
}

# Convert categorical features to numerical values
encoder = LabelEncoder()
categorical_features = {k: encoder.fit_transform([v])[0] for k, v in categorical_features.items()}

# Prepare input data
data = pd.DataFrame([[
    categorical_features["BodyType"], categorical_features["Sex"], categorical_features["Diet"], categorical_features["HowOftenShower"],
    categorical_features["HeatingEnergySource"], categorical_features["Transport"], categorical_features["VehicleType"],
    categorical_features["SocialActivity"], monthly_grocery_bill, categorical_features["FrequencyAirTravel"],
    vehicle_monthly_distance, categorical_features["WasteBagSize"], waste_bag_weekly_count, how_long_tv_pc,
    how_many_new_clothes, how_long_internet, categorical_features["EnergyEfficiency"], categorical_features["Recycling"],
    categorical_features["CookingWith"]
]], columns=[
    "BodyType", "Sex", "Diet", "HowOftenShower", "HeatingEnergySource", "Transport", "VehicleType", "SocialActivity",
    "MonthlyGroceryBill", "FrequencyAirTravel", "VehicleMonthlyDistance", "WasteBagSize", "WasteBagWeeklyCount",
    "HowLongTVPC", "HowManyNewClothes", "HowLongInternet", "EnergyEfficiency", "Recycling", "CookingWith"
])

# Prediction
if st.sidebar.button("Predict Carbon Emission"):
    prediction = model.predict(data)[0]
    st.subheader("📊 Prediction Result")
    st.write(f"Your estimated annual carbon footprint: **{prediction:.2f} kg CO₂**")
    
    # Visualization
    st.subheader("📈 Carbon Footprint Breakdown")
    breakdown = pd.DataFrame({
        "Category": ["Energy", "Transport", "Food", "Waste"],
        "Emission (kg CO₂)": [monthly_grocery_bill*0.5, vehicle_monthly_distance*0.1, how_many_new_clothes*2, waste_bag_weekly_count*3]
    })
    fig = px.pie(breakdown, names='Category', values='Emission (kg CO₂)', title="Carbon Footprint Contribution by Category")
    st.plotly_chart(fig, use_container_width=True)
