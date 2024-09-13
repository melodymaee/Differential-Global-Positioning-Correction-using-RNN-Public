from pymavlink import mavutil
connection = mavutil.mavlink_connection('COM5', baud=9600)

while True:
    msg = connection.recv_msg()

    # mode = connection.flightmode
    # print("Current Flight Mode:", mode)
    if msg is not None:
        #    if msg.get_srcComponent() == 200:
        #        if msg.get_type() == 'STATUSTEXT':
        #            print(msg)
        # if msg.get_type() == 'GLOBAL_POSITION_INT':
        print(msg)
        if msg.get_type() == 'RC_CHANNELS':
            # print(msg)
            # Retrieve the RC channel 11 value
            data = []
            rc11 = msg.chan11_raw
            rc12 = msg.chan12_raw
            rc13 = msg.chan13_raw

            # print(rc11, rc12, rc13)
