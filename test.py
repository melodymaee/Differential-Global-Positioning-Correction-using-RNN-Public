import os
os.system('sudo chmod 666 /dev/ttyTHS1')

import numpy as np
import re
import pandas as pd
import time
import socket
from collections import deque
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import GRU, Dense, TimeDistributed
# from keras.layers import SimpleRNN, Dense, TimeDistributed
# from keras.layers import LSTM, Dense, TimeDistributed
from dronekit import connect, LocationGlobalRelative, VehicleMode
from keras.layers import Bidirectional, LSTM, Dense, TimeDistributed
from pymavlink import mavutil

# Establish drone connection
vehicle = connect('/dev/ttyTHS1', baud = 921600, wait_ready=True)
print('Vehicle is connected')

# c = serial.Serial('COM7', 9600)

ref = 2

#ref1_main = lat: 8.2422474; long: 124.2440479; alt: 74.889
#ref2 = lat: 8.2423256; 124.2440685; 57.672
#ref3 = lat: 8.2423624; long:124.2439292; alt: 43.995

main_lat = 8.2422474
main_long = 124.2440479
main_alt = 74.889

ref2_lat = 8.2423256
ref2_lng = 124.2440685
ref2_alt = 74.889

ref3_lat = 8.2423624
ref3_lng = 124.2439292
ref3_alt = 74.889

# Define the model architecture
model = Sequential()
# model.add(LSTM(10, return_sequences=True, input_shape=(None, 4)))  # Adjust input shape to match the dimensionality of the input data
# model.add(SimpleRNN(8, return_sequences=True, input_shape=(None, 2)))  # Adjust input shape to match the dimensionality of the input data
model.add(Bidirectional(LSTM(8, return_sequences=True), input_shape=(None, 4)))
# model.add(GRU(12, return_sequences=True, input_shape=(None, 4)))  # Adjust input shape to match the dimensionality of the input data


model.add(TimeDistributed(Dense(4)))
model.compile(optimizer='rmsprop', loss='mse')

# Initialize a deque to store historical input-output pairs
history = deque(maxlen=10)  # Keep the last 10 input-output pairs as historical data

def get_phone_gps(sentence):
    # Define regular expression pattern to match GPRMC NMEA sentence
    pattern = r'\$GPRMC,\d+,\w,(\d+\.\d+),([NS]),(\d+\.\d+),([EW]),.*?,(\d+\.\d+),[A-Z],.*?\*.*?$'

    # Search for the pattern in the sentence
    match = re.match(pattern, sentence.decode())

    if match:
        # Extract latitude, longitude, and altitude
        latitude = float(match.group(1))
        latitude_direction = match.group(2)
        longitude = float(match.group(3))
        longitude_direction = match.group(4)
        altitude = float(match.group(5))

        # Convert latitude and longitude from degrees and decimal minutes to decimal degrees
        lat_degrees = int(latitude / 100)
        lat_minutes = latitude % 100
        lat_decimal = lat_minutes / 60
        latitude_decimal = lat_degrees + lat_decimal

        lon_degrees = int(longitude / 100)
        lon_minutes = longitude % 100
        lon_decimal = lon_minutes / 60
        longitude_decimal = lon_degrees + lon_decimal

        # Return latitude, longitude, and altitude
        return (latitude_decimal if latitude_direction == 'N' else -latitude_decimal,
                longitude_decimal if longitude_direction == 'E' else -longitude_decimal,
                74.889)
    else:
        return None, None, 74.889 # Return None values if sentence doesn't match the expected format

def gps_err_input(location):
    timestamps = pd.date_range(datetime.now(), periods=1, freq='s')
    
    for _ in range(10):  # Wait for 10 seconds
        time.sleep(1)  # Wait for 1 second
        new_lat = location.lat
        new_long = location.lon
        new_alt = location.alt
        
        if new_lat is not None and new_long is not None and new_alt is not None:
            break  # Break the loop once valid GPS coordinates are received
    
    if new_lat is None or new_long is None or new_alt is None:
        raise ValueError("Invalid GPS coordinates")
    
    err_lat = main_lat - new_lat
    err_long = main_long - new_long
    err_alt = main_alt - new_alt
    
    return timestamps, err_lat, err_long, err_alt


def gps_raw(location):
    timestamps = pd.date_range(datetime.now(), periods=1, freq='s')
    new_lat = location.lat
    new_long = location.lon
    new_alt = location.alt
    
    return timestamps, new_lat, new_long, new_alt

    

# Function to generate random GPS data
def generate_random_gps_data(num_points):
    timestamps = pd.date_range(datetime.now(), periods=num_points, freq='s')
    latitudes = np.random.uniform(low=-90, high=90, size=num_points)
    longitudes = np.random.uniform(low=-180, high=180, size=num_points)
    altitudes = np.random.uniform(low=0, high=5000, size=num_points)
    return timestamps, latitudes, longitudes, altitudes

def ref2_truth_err (lat,lng,alt):
    
    true_err_lat_ref2 = ref2_lat - lat
    true_err_lng_ref2 = ref2_lng - lng
    true_err_alt_ref2 = ref2_alt - alt
    
    return true_err_lat_ref2, true_err_lng_ref2, true_err_alt_ref2

def ref3_truth_err (lat,lng,alt):
    
    true_err_lat_ref3 = ref3_lat - lat
    true_err_lng_ref3 = ref3_lng - lng
    true_err_alt_ref3 = ref3_alt - alt
    
    return true_err_lat_ref3, true_err_lng_ref3, true_err_alt_ref3

# Define a function to train the model with historical and current data, and make predictions in real-time
def train_and_predict_real_time(model):
    while True:
        # Read a line from the serial port
        # sentence = c.readline()

        # Extract data from the received NMEA sentence
        # phone_lat, phone_lng, phone_alt = get_phone_gps(sentence)
        
        # if phone_lat is not None and phone_lng is not None:
        #     print(f"Phone Lat: {phone_lat}\tPhone Long: {phone_lng}\tPhone Alt: {phone_alt}")
        # else:
        #     print("No GPS From Phone!")
            
        # Generate real-time input data (replace this with your actual input method)
        #curr_time, curr_lat_err, curr_lng_err, curr_alt_err = gps_err_input(vehicle.location.global_frame)
        # Convert timestamps to numerical representation (in seconds since epoch)
        #curr_timestamp_numeric = (curr_time - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        
        # corrected gps from phone
        # corrected_lat = phone_lat - curr_lat_err
        # corrected_lng = phone_lng - curr_lng_err
        # corrected_alt = phone_alt - curr_alt_err
        
        # print(f"Corrected Lat: {corrected_lat}\tCorrected Long: {corrected_lng}\tCorrected Alt: {corrected_alt}")
        
        # if ref == 2:
        #     ref = 3
        #     truth_err_lat, truth_err_lng, truth_err_alt = ref2_truth_err(phone_lat, phone_lng, phone_alt)
        #     print(f"Truth Error Lat: {truth_err_lat}\tTruth Error Long: {truth_err_lng}\tTruth Error Alt: {truth_err_alt}")
            
        # elif ref == 3:
        #     ref = 2
        #     truth_err_lat, truth_err_lng, truth_err_alt = ref3_truth_err(phone_lat, phone_lng, phone_alt)
        #     print(f"Truth Error Lat: {truth_err_lat}\tTruth Error Long: {truth_err_lng}\tTruth Error Alt: {truth_err_alt}")
        
        curr_time_raw, new_lat, new_lng, new_alt = gps_raw(vehicle.location.global_frame)
        curr_timestamp_numeric = (curr_time_raw - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        
        curr_input = np.column_stack((curr_timestamp_numeric, new_lat, new_lng, new_alt)).reshape(1, 1, 4)
        # curr_output = np.column_stack((curr_timestamp_numeric, truth_err_lat, truth_err_lng, truth_err_alt)).reshape(1, 1, 4)  # This should be from the other GPS
        

        print(f"Current Time: {curr_timestamp_numeric}\tCurrent Lat: {new_lat}\tCurrent Error Long: {new_lng}\tCurrent Error Alt: {new_alt}")
        #print(f"Current Time: {curr_timestamp_numeric}\tCurrent Error Lat: {curr_lat_err}\tCurrent Error Long: {curr_lng_err}\tCurrent Error Alt: {curr_alt_err}")

        curr_output = np.column_stack((curr_timestamp_numeric, new_lat, new_lng, new_alt)).reshape(1, 1, 4)

        # Add the current input-output pair to the history
        history.append((curr_input, curr_output))
        
        # Concatenate historical input-output pairs with the current input-output pair
        training_data = np.concatenate([data[0] for data in history], axis=1)
        training_labels = np.concatenate([data[1] for data in history], axis=1)
        
        # Train the model on the combined historical and current data
        # Train the model on the combined historical and current data
        model.fit(training_data, training_labels, epochs=1, batch_size=1, verbose=0)
        
        # Print historical data
        print("Historical Data:")
        for i, (input_data, output_data) in enumerate(history):
            print("Input Data at Timestep {}: {}".format(i, input_data))
            print("Output Data at Timestep {}: {}".format(i, output_data))
        
        # Predict the output for the current input data
        predicted_error = model.predict(curr_input)
        
        # Print or handle the predictions
        print("\nCurrent Input Data:")
        print(curr_input)
        print("Predicted Output:")
        print(predicted_error)
        print("------------------------")
        
        time.sleep(5)  # Wait for 2 seconds before processing the next data point

# Start continuous training and real-time prediction
train_and_predict_real_time(model)

