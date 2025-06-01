import streamlit as st
import numpy as np
import pickle

# Save the best model
with open('best_price_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("✅ Model saved as best_price_model.pkl")

# --- Load the model and scalers ---
with open('best_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- Streamlit UI ---
st.set_page_config(page_title="Flight Price Prediction", page_icon="✈️", layout="centered")
st.title("✈️ Flight Price Prediction")

st.sidebar.header("Flight Details")
airline = st.sidebar.selectbox("Airline", ["AirAsia", "Indigo", "GO_FIRST", "SpiceJet", "Air_India", "Vistara"])
source = st.sidebar.selectbox("Source City", ["Delhi", "Hyderabad", "Bangalore", "Mumbai", "Kolkata", "Chennai"])
departure = st.sidebar.selectbox("Departure Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"])
arrival = st.sidebar.selectbox("Arrival Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"])
destination = st.sidebar.selectbox("Destination City", ["Delhi", "Hyderabad", "Mumbai", "Bangalore", "Chennai", "Kolkata"])
stops = st.sidebar.selectbox("Stops", ["zero", "one", "two_or_more"])
flight_class = st.sidebar.selectbox("Class", ["Economy", "Business"])
days_left = st.sidebar.slider("Days Left to Departure", 0, 365, 30)

# --- Mapping dictionaries (same as model training) ---
airline_dict = {'AirAsia': 0, "Indigo": 1, "GO_FIRST": 2, "SpiceJet": 3, "Air_India": 4, "Vistara": 5}
source_dict = {'Delhi': 0, "Hyderabad": 1, "Bangalore": 2, "Mumbai": 3, "Kolkata": 4, "Chennai": 5}
departure_dict = {'Early_Morning': 0, "Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4, "Late_Night": 5}
arrival_dict = {'Early_Morning': 0, "Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4, "Late_Night": 5}
destination_dict = {'Delhi': 0, "Hyderabad": 1, "Mumbai": 2, "Bangalore": 3, "Chennai": 4, "Kolkata": 5}
class_dict = {'Economy': 0, 'Business': 1}
stops_dict = {'zero': 0, "one": 1, "two_or_more": 2}

# --- Predict Button ---
if st.button("🎯 Predict Price"):
    try:
        # Prepare input array
        input_features = np.array([[
            airline_dict[airline],
            source_dict[source],
            departure_dict[departure],
            arrival_dict[arrival],
            destination_dict[destination],
            stops_dict[stops],
            class_dict[flight_class],
            days_left
        ]])

        # Scale input
        scaled_input = minmax_scaler_X.transform(input_features)

        # Predict and inverse transform
        scaled_prediction = model.predict(scaled_input)
        predicted_price = minmax_scaler_y.inverse_transform(scaled_prediction.reshape(-1, 1))[0][0]

        st.success(f"💸 Estimated Flight Price: ₹{round(predicted_price, 2)}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
