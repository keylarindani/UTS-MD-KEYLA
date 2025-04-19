import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, scaler, dan label encoder
with open("best_model_rf.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

st.title("Hotel Booking Cancellation Prediction")

# Form input
st.subheader("Masukkan informasi reservasi:")
lead_time = st.number_input("Lead Time (hari)", min_value=0)
no_of_adults = st.number_input("Jumlah Dewasa", min_value=0)
no_of_children = st.number_input("Jumlah Anak", min_value=0)
no_of_weekend_nights = st.number_input("Jumlah Malam Akhir Pekan", min_value=0)
no_of_week_nights = st.number_input("Jumlah Malam Hari Kerja", min_value=0)
type_of_meal_plan = st.selectbox("Paket Makanan", options=["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
required_car_parking_space = st.selectbox("Butuh Parkir?", options=[0, 1])
room_type_reserved = st.selectbox("Tipe Kamar", options=["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
market_segment_type = st.selectbox("Tipe Segmen Market", options=["Offline", "Online", "Corporate", "Aviation", "Complementary"])
repeated_guest = st.selectbox("Tamu Berulang?", options=[0, 1])
no_of_previous_cancellations = st.number_input("Jumlah Pembatalan Sebelumnya", min_value=0)
no_of_previous_bookings_not_canceled = st.number_input("Jumlah Booking Sukses Sebelumnya", min_value=0)
avg_price_per_room = st.number_input("Rata-rata Harga Kamar (EUR)", min_value=0.0)
no_of_special_requests = st.number_input("Jumlah Permintaan Khusus", min_value=0)

# Encode fitur kategorikal
meal_map = {"Meal Plan 1": 0, "Meal Plan 2": 1, "Meal Plan 3": 2, "Not Selected": 3}
room_map = {"Room_Type 1": 0, "Room_Type 2": 1, "Room_Type 3": 2, "Room_Type 4": 3,
            "Room_Type 5": 4, "Room_Type 6": 5, "Room_Type 7": 6}
segment_map = {"Offline": 0, "Online": 1, "Corporate": 2, "Aviation": 3, "Complementary": 4}

# Buat DataFrame dari input user
input_data = pd.DataFrame([[
    lead_time,
    no_of_adults,
    no_of_children,
    no_of_weekend_nights,
    no_of_week_nights,
    meal_map[type_of_meal_plan],
    required_car_parking_space,
    room_map[room_type_reserved],
    segment_map[market_segment_type],
    repeated_guest,
    no_of_previous_cancellations,
    no_of_previous_bookings_not_canceled,
    avg_price_per_room,
    no_of_special_requests
]])

# Scaling
input_scaled = scaler.transform(input_data)

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("Booking kemungkinan akan DIBATALKAN ❌")
    else:
        st.success("Booking kemungkinan TIDAK dibatalkan ✅")
