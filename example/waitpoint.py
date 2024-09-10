from time import sleep
import xpc

def ex():
    print("X-Plane Connect example script")
    print("Setting up simulation")
    with xpc.XPlaneConnect() as client:
        # Verify connection
        try:
            # If X-Plane does not respond to the request, a timeout error
            # will be raised.
            client.getDREF("sim/test/test_float")
        except:
            print("Error establishing connection to X-Plane.")
            print("Exiting...")
            return

        # Set position of the player aircraft
        print("Setting position")
        #       Lat     Lon         Alt   Pitch Roll Yaw Gear
        posi = [37.524, -122.06899, 2500, 0,    0,   0,  0]
        client.sendPOSI(posi)
        
        # Set position of a non-player aircraft
        print("Setting NPC position")
        #       Lat       Lon         Alt   Pitch Roll Yaw Gear
        posi = [37.52465, -122.06899, 2500, 0,    20,   0,  0]
        client.sendPOSI(posi, 1)

        # Set angle of attack, velocity, and orientation using the DATA command(*(()))
        # print("Setting orientation")
        data = [\
            [18,   0, -998,   0, -998, -998, -998, -998, -998],\
            [ 3, 130,  130, 130,  130, -998, -998, -998, -998],\
            [16,   0,    0,   0, -998, -998, -998, -998, -998]\
            ]
        client.sendDATA(data)
        # client.sendDATA(data, 1)

        # Set control surfaces and throttle of the player aircraft using sendCTRL
        print("Setting controls")
        ctrl = [0.0, 0.0, 0.0, 0.8]
        client.sendCTRL(ctrl)
        print("Setting NPC controls")
        ctrl = [0.0, 0.0, 0.0, 0.8]
        client.sendCTRL(ctrl, 1)

        # Define waypoints as [latitude, longitude, altitude]
        # waypoints = [
        #     [37.7749, -122.4194, 1000],  # San Francisco, 1000 ft
        #     [34.0522, -118.2437, 2000],  # Los Angeles, 2000 ft
        #     [36.1699, -115.1398, 1500]   # Las Vegas, 1500 ft
        # ]
        waypoints = [ 37, -122, 2500]
        # Send the waypoints to the aircraft with ID 1
        # client.sendWYPT(0, waypoints)
        client.sendWYPT(3, [])
        print(client.sendWYPT(1, waypoints))
        client.close()
        # # Pause the sim
        # print("Pausing")
        # client.pauseSim(True)
        # sleep(2)

        # # Toggle pause state to resume
        # print("Resuming")
        # client.pauseSim(False)

        # # Stow landing gear using a dataref
        # print("Stowing gear")
        # gear_dref = "sim/cockpit/switches/gear_handle_status"
        # client.sendDREF(gear_dref, 0)

        # # Let the sim run for a bit.
        # sleep(4)

        # # Make sure gear was stowed successfully
        # gear_status = client.getDREF(gear_dref)
        # if gear_status[0] == 0:
        #     print("Gear stowed")
        # else:
        #     print("Error stowing gear")

        print("End of Python client example")
        input("Press any key to exit...")

def test_sendWYPT():
    # Setup
    client = xpc.XPlaneConnect()
    points = [\
        37.5245, -122.06899, 2500,\
        37.455397, -122.050037, 2500,\
        37.469567, -122.051411, 2500,\
        37.479376, -122.060509, 2300,\
        37.482237, -122.076130, 2100,\
        37.474881, -122.087288, 1900,\
        37.467660, -122.079391, 1700,\
        37.466298, -122.090549, 1500,\
        37.362562, -122.039223, 1000,\
        37.361448, -122.034416, 1000,\
        37.361994, -122.026348, 1000,\
        37.365541, -122.022572, 1000,\
        37.373727, -122.024803, 1000,\
        37.403869, -122.041283, 50,\
        37.418544, -122.049222, 6]

    # Execution
    client.sendPOSI([37.5245, -122.06899, 2500])
    client.sendWYPT(3, [])
    client.sendWYPT(1, points)
    # NOTE: Manually verify that points appear on the screen in X-Plane

    # Cleanup
    client.close()


if __name__ == "__main__":
    # ex()
    test_sendWYPT()