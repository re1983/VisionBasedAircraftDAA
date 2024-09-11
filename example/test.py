import xpc
import time

client = xpc.XPlaneConnect()

# Set the initial position for the player aircraft
def set_player_position():
    initial_lat = 37.7749  # Example latitude for San Francisco
    initial_lon = -122.4194  # Example longitude
    initial_alt = 2500  # Altitude in feet
    heading = 0.0  # Heading north (0 degrees)
    pos = [initial_lat, initial_lon, initial_alt, heading, 0.0, 0.0, 1]  # Last value 1 indicates player aircraft
    client.sendPOSI(pos)

# Set control to fly the player aircraft straight north (using heading and throttle)
def fly_player_north():
    ctrl = [0.0, 0.0, 0.0, 0.8]  # No roll, no pitch, no yaw, throttle 80%
    client.sendCTRL(ctrl)

# Set player aircraft's initial position
set_player_position()

# Start flying north after setting the position
fly_player_north()

# Simulate flying for 10 seconds
time.sleep(10)

# Set the non-player aircraft's position to move from west to east
def fly_non_player_west_to_east(start_lat, start_lon, start_alt):
    for i in range(10):  # Simulate 10 position updates
        new_lon = start_lon + 0.01 * i  # Increment longitude to move east
        pos = [start_lat, new_lon, start_alt, 0.0, 0.0, 0.0, 1]
        client.sendPOSI(pos, ac=1)  # 'ac=1' refers to non-player aircraft
        time.sleep(1)

# Example coordinates for non-player aircraft starting west of player aircraft
fly_non_player_west_to_east(37.7749, -123.0, 2500)

# Set camera view to follow the player aircraft
# client.sendVIEW(85)  # 85 is the external view around the player aircraft
