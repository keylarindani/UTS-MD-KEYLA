import streamlit as st
import pandas as pd
import numpy as np
import pickle
import zipfile

# Ekstrak model dari ZIP
with zipfile.ZipFile("best_model_rf.zip", "r") as zip_ref:
    zip_ref.extractall()

# Load model dan scaler
with open("best_model_rf.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler (1).pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Prediksi Pembatalan Booking Hotel")

# Input user
lead_time = st.number_input("Lead Time (hari)", min_value=0)
no_of_adults = st.number_input("Jumlah Dewasa", min_value=0)
no_of_children = st.number_input("Jumlah Anak", min_value=0)
no_of_weekend_nights = st.number_input("Malam Akhir Pekan", min_value=0)
no_of_week_nights = st.number_input("Malam Hari Kerja", min_value=0)
meal_plan = st.selectbox("Paket Makanan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
parking = st.selectbox("Butuh Parkir?", [0, 1])
room_type = st.selectbox("Tipe Kamar", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
market_segment = st.selectbox("Segmen Market", ["Offline", "Online", "Corporate", "Aviation", "Complementary"])
repeated_guest = st.selectbox("Tamu Berulang?", [0, 1])
cancel_before = st.number_input("Pembatalan Sebelumnya", min_value=0)
success_before = st.number_input("Booking Sukses Sebelumnya", min_value=0)
price = st.number_input("Harga Rata-Rata Kamar", min_value=0.0)
special_request = st.number_input("Jumlah Permintaan Khusus", min_value=0)

# Mapping categorical
meal_map = {"Meal Plan 1": 0, "Meal Plan 2": 1, "Meal Plan 3": 2, "Not Selected": 3}
room_map = {
    "Room_Type 1": 0, "Room_Type 2": 1, "Room_Type 3": 2,
    "Room_Type 4": 3, "Room_Type 5": 4, "Room_Type 6": 5, "Room_Type 7": 6
}
segment_map = {
    "Offline": 0, "Online": 1, "Corporate": 2, "Aviation": 3, "Complementary": 4
}

# Pastikan nama kolom sesuai saat training
columns = [
    'lead_time', 'no_of_adults', 'no_of_children',
    'no_of_weekend_nights', 'no_of_week_nights',
    'type_of_meal_plan', 'required_car_parking_space',
    'room_type_reserved', 'market_segment_type', 'repeated_guest',
    'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
    'avg_price_per_room', 'no_of_special_requests'
]

# Buat DataFrame input
input_df = pd.DataFrame([[
    lead_time, no_of_adults, no_of_children,
    no_of_weekend_nights, no_of_week_nights,
    meal_map[meal_plan], parking, room_map[room_type],
    segment_map[market_segment], repeated_guest,
    cancel_before, success_before, price, special_request
]], columns=columns)

# Urutkan ulang kolom sesuai fitur scaler
input_df = input_df[scaler.feature_names_in_]

# Scaling
input_scaled = scaler.transform(input_df)

# Prediksi
if st.button("Prediksi"):
    result = model.predict(input_scaled)[0]
    if result == 1:
        st.error("❌ Booking kemungkinan AKAN DIBATALKAN.")
    else:
        st.success("✅ Booking kemungkinan TIDAK dibatalkan.")
