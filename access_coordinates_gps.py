
import os
os.system('sudo chmod 666 /dev/ttyTHS1')

import threading
from dronekit import connect, LocationGlobalRelative
import time

vehicle = None

# Establish drone connection

vehicle = connect('/dev/ttyTHS1', baud = 921600, wait_ready=True)
#vehicle = connect('/dev/ttyTHS1', baud = 57600, wait_ready=True)
print('Vehicle is connected')

# Function to continuously print original GPS coordinates
def print_original_gps_coordinates():
    while True:
        original_gps = vehicle.location.global_frame
        print("Original GPS Coordinates: Lat {}, Lon {}".format(original_gps.lat, original_gps.lon))
        time.sleep(1)

# Main
time.sleep(5)

# Start printing original GPS coordinates in a separate thread
original_gps_thread = threading.Thread(target=print_original_gps_coordinates)
original_gps_thread.start()

# Wait for the original GPS thread to finish
original_gps_thread.join()

vehicle.close()
