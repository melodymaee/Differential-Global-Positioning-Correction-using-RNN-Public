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
from keras.layers import SimpleRNN, Dense, TimeDistributed
from keras.layers import LSTM, Dense, TimeDistributed
from dronekit import connect, LocationGlobalRelative, VehicleMode
from keras.layers import Bidirectional, LSTM, Dense, TimeDistributed
from pymavlink import mavutil
import socket
import csv

HOST = '0.0.0.0'
PORT = 65432

reference_points = {
    1: (1, 8.2422474, 124.2440479, 74.889),
    2: (2, 8.2422474, 124.2440479, 74.889),
    3: (3, 8.2423624, 124.2439292, 74.889),
    4: (4, 8.2422245, 124.2439658, 74.889),
    7: (7, 8.2422803, 124.2439593, 74.889),
    10: (10, 8.2422347, 124.2440257, 74.889)
}

main_lat = 8.2422474
main_lng = 124.2440479
main_alt = 74.889

# Define the model architecture
model = Sequential()
#model.add(LSTM(10, return_sequences=True, input_shape=(None, 4)))  # Adjust input shape to match the dimensionality of the input data
#model.add(SimpleRNN(5, return_sequences=True, input_shape=(None, 4)))  # Adjust input shape to match the dimensionality of the input data
#model.add(GRU(12, return_sequences=True, input_shape=(None, 4)))  # Adjust input shape to match the dimensionality of the input data
model.add(Bidirectional(LSTM(8, return_sequences=True), input_shape=(None, 4)))
model.add(TimeDistributed(Dense(4)))
model.compile(optimizer='rmsprop', loss='mse')

# Initialize a deque to store historical input-output pairs
history = deque(maxlen=1000)  # Keep the last 10 input-output pairs as historical data

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


def gps_raw(location):
    #timestamps = pd.date_range(datetime.now(), periods=1, freq='s')
    new_lat = location.lat
    new_long = location.lon
    new_alt = location.alt
    
    return new_lat, new_long, new_alt
    
# Establish drone connection
vehicle = connect('/dev/ttyTHS1', baud = 921600, wait_ready=True)
print('Vehicle is connected')
        
# Open a CSV file in write mode and create a CSV writer object
with open('received_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['timestamp','corr_lat', 'corr_lng', 'corr_alt','true_lat', 'true_lng', 'true_alt', 'ref_num']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write the header row to the CSV file
    writer.writeheader()        

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
        # print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                
                received_data = data.decode().split(",")  # Split the received data by comma
                ref_num = float(received_data[0])
                phone_lat = float(received_data[1])
                phone_lng = float(received_data[2])
                phone_alt = float(received_data[3])
                phone_err_lat = float(received_data[4])
                phone_err_lng = float(received_data[5])
                phone_err_alt = float(received_data[6])
                
                _, true_lat, true_lng, true_alt = reference_points[ref_num]
                
                base_lat, base_lng, base_alt = gps_raw(vehicle.location.global_frame)
                #curr_timestamp = (curr_time_raw - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
                
                #print(curr_timestamp,base_lat,base_lng,base_alt)
                
                err_lat = base_lat - main_lat
                err_lng = base_lng - main_lng
                err_alt = base_alt - main_alt
                
                # Corrected based on base_gps error
                corr_lat = phone_lat - err_lat
                corr_lng = phone_lng - err_lng
                corr_alt = phone_alt - err_alt
                
                now = datetime.now()
                timestamp = int(now.timestamp())
                
                err_input = np.column_stack((timestamp, err_lat, err_lng, err_alt)).reshape(1, 1, 4)
                err_output = np.column_stack((timestamp, phone_err_lat, phone_err_lng, phone_err_alt)).reshape(1, 1, 4)
                
                # Add the current input-output pair to the history
                history.append((err_input, err_output))
                
                # Concatenate historical input-output pairs with the current input-output pair
                training_data = np.concatenate([data[0] for data in history], axis=1)
                training_labels = np.concatenate([data[1] for data in history], axis=1)
                
                # Train the model on the combined historical and current data
                # Train the model on the combined historical and current data
                model.fit(training_data, training_labels, epochs=1, batch_size=1, verbose=0)
                
                # Write the received data to the CSV file
                writer.writerow({'timestamp': datetime.now(),'corr_lat': corr_lat, 'corr_lng': corr_lng, 'corr_alt': corr_alt,'true_lat': true_lat, 'true_lng': true_lng, 'true_alt': true_alt, 'ref_num': ref_num})
                
                # Print historical data
                print("Historical Data:")
                
                for i, (input_data, output_data) in enumerate(history):
                    print("Input Data at Timestep {}: {}".format(i, input_data))
                    print("Output Data at Timestep {}: {}".format(i, output_data))
