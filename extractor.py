'''from serial import serial
import re

# CC:F8:26:E4:65:B1

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

# Open the serial port
#c = s.Serial('/dev/ttyTHS1', 9600)
ser = serial.serial('/dev/ttyS0', 115200)

try:
    while True:
        # Read a line from the serial port
        sentence = ser.readline()

        # Extract data from the received NMEA sentence
        latitude, longitude, altitude = get_phone_gps(sentence)
        if latitude is not None and longitude is not None and altitude is not None:
            print("Latitude:", latitude)
            print("Longitude:", longitude)
            print("Longitude:", altitude)
        
except KeyboardInterrupt:
    print("Exiting...")

finally:
    # Close the serial port
    c.close()'''


import serial
import re

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

# Open the serial port
ser = serial.Serial('/dev/ttyS', 115200)

try:
    while True:
        # Read a line from the serial port
        sentence = ser.readline()

        # Extract data from the received NMEA sentence
        latitude, longitude, altitude = get_phone_gps(sentence)
        if latitude is not None and longitude is not None and altitude is not None:
            print("Latitude:", latitude)
            print("Longitude:", longitude)
            print("Longitude:", altitude)
        
except KeyboardInterrupt:
    print("Exiting...")

finally:
    # Close the serial port
    ser.close()

