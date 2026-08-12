import streamlit as st
import pandas as pd
import joblib


# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# 2. LOAD MODEL AND PIPELINE

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    pipeline = joblib.load("pipeline.pkl")

    return model, pipeline


model, pipeline = load_model()


# 3. TITLE

st.title("🏠 California House Price Predictor")

st.write(
    "Enter the location and housing information "
    "to predict the median house value."
)

st.divider()


# 4. LOCATION

st.header("📍 Location")

col1, col2, col3 = st.columns(3)

with col1:
    longitude = st.slider(
        "Longitude",
        min_value=-125.0,
        max_value=-114.0,
        value=-119.5,
        step=0.01
    )

with col2:
    latitude = st.slider(
        "Latitude",
        min_value=32.0,
        max_value=42.0,
        value=35.5,
        step=0.01
    )

with col3:
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
    )


st.divider()


# 5. HOUSING INFORMATION

st.header("🏡 Housing Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Property")

    housing_median_age = st.number_input(
        "Housing Median Age",
        min_value=1.0,
        max_value=100.0,
        value=29.0,
        step=1.0
    )

    total_rooms = st.number_input(
        "Total Rooms",
        min_value=1.0,
        max_value=50000.0,
        value=2600.0,
        step=1.0
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        min_value=1.0,
        max_value=10000.0,
        value=530.0,
        step=1.0
    )


with col2:
    st.subheader("👨‍👩‍👧‍👦 Population")

    population = st.number_input(
        "Population",
        min_value=1.0,
        max_value=40000.0,
        value=1400.0,
        step=1.0
    )

    households = st.number_input(
        "Households",
        min_value=1.0,
        max_value=10000.0,
        value=500.0,
        step=1.0
    )

    median_income_dollars = st.number_input(
        "Median Household Income ($)",
        min_value=10000.0,
        max_value=200000.0,
        value=35000.0,
        step=1000.0
    )


st.divider()


# 6. PREDICTION

st.header("🔮 Prediction")

if st.button(
    "🏠 Predict House Price",
    type="primary",
    use_container_width=True
):

    # Check inputs
    if total_bedrooms > total_rooms:
        st.error("Total bedrooms cannot be greater than total rooms.")

    elif households > population:
        st.error("Households cannot be greater than population.")

    else:
        # Convert income from dollars to the format used by the dataset
        median_income = median_income_dollars / 10000

        input_data = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity
        }])

        # Preprocess the input
        transformed_data = pipeline.transform(input_data)

        # Make prediction
        prediction = model.predict(transformed_data)[0]

        # Display prediction
        st.success("✅ Prediction complete!")

        st.metric(
            "🏠 Estimated Median House Value",
            f"${prediction:,.0f}"
        )


# 7. FOOTER

st.divider()

st.caption(
    "🏠 California Housing Price Prediction · "
    "XGBoost Regressor"
)