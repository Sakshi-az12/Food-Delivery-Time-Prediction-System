import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Page Configuration
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🍔",
    layout="centered"
)

# Title
st.title("🍔 Food Delivery Time Prediction System")
st.write("Predict the estimated food delivery time using Machine Learning.")

# Load Dataset
df = pd.read_csv("cleaned_data.csv")

# Fill Missing Values (Safety)
df['Courier_Experience_yrs'] = df['Courier_Experience_yrs'].fillna(
    df['Courier_Experience_yrs'].median()
)

# Features and Target
X = df.drop("Delivery_Time_min", axis=1)
y = df["Delivery_Time_min"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

# Sidebar
st.sidebar.header("Project Information")
st.sidebar.write("""
**Algorithm:** Linear Regression

**Dataset:** Food Delivery Dataset

**Developer:** Sakshi Waghmare
""")

# User Input
st.subheader("Enter Order Details")

distance = st.number_input("Distance (km)", min_value=0.0, value=8.5)

weather = st.selectbox(
    "Weather",
    [0,1,2,3,4],
    help="0=Cloudy, 1=Foggy, 2=Rainy, 3=Stormy, 4=Sunny"
)

traffic = st.selectbox(
    "Traffic Level",
    [0,1,2],
    help="0=Low, 1=Medium, 2=High"
)

time_day = st.selectbox(
    "Time of Day",
    [0,1,2,3],
    help="0=Morning, 1=Afternoon, 2=Evening, 3=Night"
)

vehicle = st.selectbox(
    "Vehicle Type",
    [0,1,2],
    help="0=Bike, 1=Scooter, 2=Car"
)

prep_time = st.number_input(
    "Preparation Time (minutes)",
    min_value=1,
    value=15
)

experience = st.number_input(
    "Courier Experience (Years)",
    min_value=0.0,
    value=3.0
)

# Prediction
if st.button("Predict Delivery Time"):

    data = [[
        distance,
        weather,
        traffic,
        time_day,
        vehicle,
        prep_time,
        experience
    ]]

    prediction = model.predict(data)

    st.success("Prediction Completed Successfully!")

    st.metric(
        label="Estimated Delivery Time",
        value=f"{prediction[0]:.2f} Minutes"
    )

    st.write(f"### Model Accuracy (R² Score): {accuracy:.3f}")

    st.balloons()