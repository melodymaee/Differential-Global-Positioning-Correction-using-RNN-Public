import tensorflow as tf
from tensorflow import keras
from dronekit import connect, LocationGlobalRelative
from pymavlink import mavutil
import time
import random
import threading

# Establish drone connection
vehicle = connect('/dev/ttyTHS1', baud = 921600, wait_ready=True)
print('Vehicle is connected')

# vehicle = connect('/dev/ttyAMA0', baud=921600, wait_ready=True)
# print('Vehicle is ready and armed.')


# Define a feed-forward neural network model
def create_model():
    model = keras.Sequential([
       
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2)  # Output layer with 2 units for latitude and longitude
    ])
    model.compile(optimizer='adam', loss='mse')  # Using mean squared error loss
    return model

# Function to add simulated DGPS error to GPS coordinates
def add_dgps_error(location):
    lat_offset = random.uniform(-0.0001, 0.0001)
    lon_offset = random.uniform(-0.0001, 0.0001)
    new_lat = location.lat + lat_offset
    new_lon = location.lon + lon_offset
    return LocationGlobalRelative(new_lat, new_lon, location.alt)

# Function to continuously collect noisy GPS coordinates
def collect_noisy_gps_coordinates():
    while True:
        yield add_dgps_error(vehicle.location.global_frame)
        time.sleep(1)

# Function to preprocess data for training
def preprocess_data(gps_data):
    return [[gps.lat, gps.lon] for gps in gps_data]


# Main training loop with manual backpropagation
def train_model(model, gps_data):
    optimizer = tf.keras.optimizers.Adam()  # Define optimizer
    
    for _ in range(NUM_TRAINING_STEPS):
        noisy_gps = next(gps_data)
        noisy_gps_array = preprocess_data([noisy_gps])
        target_gps = [[noisy_gps.lat, noisy_gps.lon]]

        with tf.GradientTape() as tape:
            # Forward pass
            predicted_gps = model(noisy_gps_array, training=True)
            # Compute loss
            loss = tf.keras.losses.mean_squared_error(target_gps, predicted_gps)

        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        # Update weights
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        time.sleep(1)


# # Main training loop
# def train_model(model, gps_data):
#     for _ in range(NUM_TRAINING_STEPS):
#         noisy_gps = next(gps_data)
#         noisy_gps_array = preprocess_data([noisy_gps])
#         corrected_gps = model.predict(noisy_gps_array)
#         target_gps = [[noisy_gps.lat, noisy_gps.lon]]
#         model.train_on_batch(noisy_gps_array, target_gps)
#         time.sleep(1)

# # Function to continuously print corrected GPS coordinates
# def print_corrected_gps_coordinates(model, gps_data):
#     while True:
#         noisy_gps = next(gps_data)
#         noisy_gps_array = preprocess_data([noisy_gps])
#         corrected_gps = model.predict(noisy_gps_array)[0]
#         corrected_gps_location = LocationGlobalRelative(corrected_gps[0], corrected_gps[1], noisy_gps.alt)
#         print("Corrected GPS Coordinates:", corrected_gps_location)
#         time.sleep(1)
        

# Function to continuously print noisy and corrected GPS coordinates
def print_gps_coordinates(model, gps_data):
    while True:
        noisy_gps = next(gps_data)
        noisy_gps_array = preprocess_data([noisy_gps])
        corrected_gps = model.predict(noisy_gps_array)[0]
        corrected_gps_location = LocationGlobalRelative(corrected_gps[0], corrected_gps[1], noisy_gps.alt)
        print("Noisy GPS Coordinates:", noisy_gps)
        print("Corrected GPS Coordinates:", corrected_gps_location)
        print("----------------------------------")
        time.sleep(1)

# Main
time.sleep(5)

# Constants
NUM_TRAINING_STEPS = 1000  # Number of training steps

# Generate noisy GPS coordinates
noisy_gps_data = collect_noisy_gps_coordinates()

# Create the model
model = create_model()

# Start training the model in a separate thread
training_thread = threading.Thread(target=train_model, args=(model, noisy_gps_data))
training_thread.start()

# Start printing GPS coordinates in a separate thread
gps_print_thread = threading.Thread(target=print_gps_coordinates, args=(model, noisy_gps_data))
gps_print_thread.start()

# Wait for both threads to finish
training_thread.join()
gps_print_thread.join()

vehicle.close()






# import tensorflow as tf
# from tensorflow import keras
# from dronekit import connect, LocationGlobalRelative
# from pymavlink import mavutil
# import time
# import random
# import threading

# # Establish drone connection
# vehicle = connect('udpin:localhost:14551', wait_ready=True)
# print('Vehicle is connected')

# # Define a feed-forward neural network model
# def create_model():
#     model = keras.Sequential([
#         keras.layers.Dense(64, activation='relu', input_shape=(2,)),
#         keras.layers.Dense(64, activation='relu'),
#         keras.layers.Dense(2)  # Output layer with 2 units for latitude and longitude
#     ])
#     model.compile(optimizer='adam', loss='mse')  # Using mean squared error loss
#     return model

# # Function to add simulated DGPS error to GPS coordinates
# def add_dgps_error(location):
#     lat_offset = random.uniform(-0.0001, 0.0001)
#     lon_offset = random.uniform(-0.0001, 0.0001)
#     new_lat = location.lat + lat_offset
#     new_lon = location.lon + lon_offset
#     return LocationGlobalRelative(new_lat, new_lon, location.alt)

# # Function to continuously collect noisy GPS coordinates
# def collect_noisy_gps_coordinates():
#     while True:
#         yield add_dgps_error(vehicle.location.global_frame)
#         time.sleep(1)

# # Function to preprocess data for training
# def preprocess_data(gps_data):
#     return [[gps.lat, gps.lon] for gps in gps_data]

# # Function to continuously print corrected GPS coordinates
# def print_corrected_gps_coordinates(model, gps_data):
#     while True:
#         noisy_gps = next(gps_data)
#         noisy_gps_array = preprocess_data([noisy_gps])
#         corrected_gps = model.predict(noisy_gps_array)[0]
#         corrected_gps_location = LocationGlobalRelative(corrected_gps[0], corrected_gps[1], noisy_gps.alt)
#         print("Corrected GPS Coordinates:", corrected_gps_location)
#         time.sleep(1)

# # Main
# time.sleep(5)

# # Generate noisy GPS coordinates
# noisy_gps_data = collect_noisy_gps_coordinates()

# # Create and train the model
# model = create_model()
# # Assuming you have a dataset for training, you can replace `None` with your training data
# # model.fit(None, None, epochs=10)

# # Start printing corrected GPS coordinates in a separate thread
# corrected_gps_thread = threading.Thread(target=print_corrected_gps_coordinates, args=(model, noisy_gps_data))
# corrected_gps_thread.start()

# # Wait for the corrected GPS thread to finish
# corrected_gps_thread.join()

# vehicle.close()
