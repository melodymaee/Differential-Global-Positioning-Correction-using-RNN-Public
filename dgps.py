from dronekit import connect, LocationGlobalRelative
from pymavlink import mavutil
import time
import random

vehicle = None

# Establish drone connection
vehicle = connect('udpin:localhost:14551', wait_ready=True)
print('Vehicle is connected')

# vehicle = connect('/dev/ttyAMA0', baud=921600, wait_ready=True)
# print('Vehicle is ready and armed.')

# Function to add simulated DGPS error to GPS coordinates
def add_dgps_error(location):
    # Simulate random error in latitude and longitude (you can adjust these values)
    lat_offset = random.uniform(-0.0001, 0.0001)  # Adjust as needed
    lon_offset = random.uniform(-0.0001, 0.0001)  # Adjust as needed
    
    # Apply the offsets to the coordinates
    new_lat = location.lat + lat_offset
    new_lon = location.lon + lon_offset
    
    # Create a new LocationGlobal object with the modified coordinates
    return LocationGlobalRelative(new_lat, new_lon, location.alt)

# Function to correct GPS coordinates based on the simulated error
def correct_gps_with_error(location, error):
    corrected_lat = location.lat - error.lat
    corrected_lon = location.lon - error.lon
    return LocationGlobalRelative(corrected_lat, corrected_lon, location.alt)

# Function to continuously print corrected GPS coordinates
def print_corrected_gps_coordinates():
    # Get initial error
    initial_error = add_dgps_error(LocationGlobalRelative(0, 0, 0))
    while True:
        gps_with_error = add_dgps_error(vehicle.location.global_frame)
        corrected_gps = correct_gps_with_error(gps_with_error, initial_error)
        print("Corrected GPS Coordinates:", corrected_gps)
        time.sleep(1)

# Main
time.sleep(5)

# Start printing corrected GPS coordinates in a separate thread
import threading
corrected_gps_thread = threading.Thread(target=print_corrected_gps_coordinates)
corrected_gps_thread.start()

# Wait for the corrected GPS thread to finish
corrected_gps_thread.join()

vehicle.close()




# from dronekit import connect, LocationGlobalRelative
# import time
# import random
# import threading
# import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from sklearn.preprocessing import MinMaxScaler

# vehicle = None
# historical_gps_data = []  # Initialize an empty historical GPS data list

# # Establish drone connection
# vehicle = connect('udpin:localhost:14551', wait_ready=True)
# print('Vehicle is connected')

# # Function to add simulated DGPS error to GPS coordinates
# def add_dgps_error(location):
#     # Simulate random error in latitude and longitude
#     lat_offset = random.uniform(-0.0001, 0.0001)
#     lon_offset = random.uniform(-0.0001, 0.0001)
#     new_lat = location.lat + lat_offset
#     new_lon = location.lon + lon_offset
#     return LocationGlobalRelative(new_lat, new_lon, location.alt)

# # Function to correct GPS coordinates based on the simulated error
# def correct_gps_with_error(location, error):
#     corrected_lat = location.lat - error.lat
#     corrected_lon = location.lon - error.lon
#     return LocationGlobalRelative(corrected_lat, corrected_lon, location.alt)

# # Define window size for input sequences
# window_size = 10

# # Define the RNN model
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, 2)))  # 2 for latitude and longitude
# model.add(LSTM(units=50, return_sequences=False))
# model.add(Dense(units=2))  # Output layer with 2 units for latitude and longitude
# model.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=True)  # Add run_eagerly=True

# # Function to preprocess data for model training
# def preprocess_data(data):
#     try:
#         if len(data) < window_size:
#             print("Not enough data for preprocessing.")
#             return None, None, None

#         scaler = MinMaxScaler(feature_range=(0, 1))
#         scaled_data = scaler.fit_transform(data)
#         X_train, y_train = [], []
#         for i in range(len(scaled_data) - window_size):
#             X_train.append(scaled_data[i:i+window_size])
#             y_train.append(scaled_data[i+window_size])
        
#         if len(X_train) == 0:
#             print("Preprocessed dataset is empty.")
#             return None, None, None
        
#         return np.array(X_train), np.array(y_train), scaler
#     except Exception as e:
#         print("Error in data preprocessing:", e)
#         return None, None, None

# # Main function for real-time data collection and model training
# def real_time_learning():
#     global historical_gps_data
#     while True:
#         # Collect new GPS data from the drone
#         new_gps_data = (vehicle.location.global_relative_frame.lat, vehicle.location.global_relative_frame.lon)
        
#         # Append new data to historical dataset
#         historical_gps_data.append(new_gps_data)

#         # If enough data points are available, preprocess and train the model
#         if len(historical_gps_data) >= window_size:
#             X_train, y_train, scaler = preprocess_data(historical_gps_data)
#             if X_train is not None:
#                 model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)  # Online learning with 1 epoch

#                 # Make real-time predictions based on the latest available data
#                 recent_gps_data = historical_gps_data[-window_size:]
#                 input_sequence = np.array(recent_gps_data).reshape(1, window_size, 2)  # Correctly reshape input data
#                 input_sequence = scaler.transform(input_sequence)  # Apply scaling
#                 predicted_position = model.predict(input_sequence)
#                 print("Predicted Position:", predicted_position)

#         # Simulate a delay before collecting the next data point
#         time.sleep(1)


# # Start real-time learning in a separate thread
# real_time_learning_thread = threading.Thread(target=real_time_learning)
# real_time_learning_thread.start()

# # Wait indefinitely
# real_time_learning_thread.join()

# # Close vehicle connection
# vehicle.close()
