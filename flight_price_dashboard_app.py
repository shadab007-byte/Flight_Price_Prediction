import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('best_price_model.pkl')

# Mapping dictionaries
airline_dict = {'AirAsia': 0, "Indigo": 1, "GO_FIRST": 2, "SpiceJet": 3, "Air_India": 4, "Vistara": 5}
source_dict = {'Delhi': 0, "Hyderabad": 1, "Bangalore": 2, "Mumbai": 3, "Kolkata": 4, "Chennai": 5}
departure_dict = {'Early_Morning': 0, "Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4, "Late_Night": 5}
arrival_dict = {'Early_Morning': 0, "Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4, "Late_Night": 5}
destination_dict = {'Delhi': 0, "Hyderabad": 1, "Mumbai": 2, "Bangalore": 3, "Chennai": 4, "Kolkata": 5}
class_dict = {'Economy': 0, 'Business': 1}
stops_dict = {'zero': 0, "one": 1, "two_or_more": 2}

# App UI
st.set_page_config(page_title="Flight Price Estimator", page_icon="✈️", layout="centered")
st.title("✈️ Flight Price Estimator")

st.markdown("""
<style>
    .main {background-color: #F0F8FF;}
    h1 {color: #333333; text-align: center;}
    .stButton button {background-color: #4682B4; color: white; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Sidebar for input
st.sidebar.header("📝 Enter Flight Details")
airline = st.sidebar.selectbox("Airline", list(airline_dict.keys()))
source = st.sidebar.selectbox("Source City", list(source_dict.keys()))
departure = st.sidebar.selectbox("Departure Time", list(departure_dict.keys()))
arrival = st.sidebar.selectbox("Arrival Time", list(arrival_dict.keys()))
destination = st.sidebar.selectbox("Destination City", list(destination_dict.keys()))
stops = st.sidebar.selectbox("Stops", list(stops_dict.keys()))
flight_class = st.sidebar.selectbox("Class", list(class_dict.keys()))
days_left = st.sidebar.slider("Days Left to Departure", 0, 365, 30)

# Prediction
if st.sidebar.button("🎯 Predict Price"):
    input_df = pd.DataFrame({
        'airline': [airline_dict[airline]],
        'source_city': [source_dict[source]],
        'departure_time': [departure_dict[departure]],
        'arrival_time': [arrival_dict[arrival]],
        'destination_city': [destination_dict[destination]],
        'stops': [stops_dict[stops]],
        'class': [class_dict[flight_class]],
        'days_left': [days_left]
    })

    predicted_price = model.predict(input_df)[0]
    lower = predicted_price * 0.9
    upper = predicted_price * 1.1

    st.success(f"💸 Estimated Flight Price: ₹{round(predicted_price, 2)}")
    st.info(f"Estimated Range: ₹{round(lower, 2)} - ₹{round(upper, 2)}")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
