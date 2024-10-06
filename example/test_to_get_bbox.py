import sys
import xpc
import PID
from datetime import datetime, timedelta
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.Qt import QtWidgets
import time
import cv2
import pymap3d as pm
import numpy as np
import win_capturewindow as wcw

window_title = 'X-System'

camera_DREFs = ["sim/cockpit2/camera/camera_offset_heading",
        "sim/cockpit2/camera/camera_offset_pitch",
        "sim/cockpit2/camera/camera_offset_roll"]
dref_heading = "sim/cockpit2/camera/camera_offset_heading"
dref_pitch = "sim/cockpit2/camera/camera_offset_pitch"
dref_roll = "sim/cockpit2/camera/camera_offset_roll"
dref_view_roll = "sim/graphics/view/view_roll"
dome_offset_heading = "sim/graphics/view/dome_offset_heading"
dome_offset_pitch = "sim/graphics/view/dome_offset_pitch"

class Aircraft:
    """Object for storing positional information for Aircraft"""
    
    def __init__(self, ac_num, east, north, up, heading, pitch=-998, roll=-998, gear=-998):
        self.id = ac_num
        self.e = east
        self.n = north
        self.u = up
        self.h = heading
        self.p = pitch
        self.r = roll
        self.g = gear
    
    def __str__(self):
        out = "Craft: %.2f, East: %.2f, North: %.2f, Up: %.2f, Heading: %.2f, Pitch: %.2f, Roll: %.2f, Gear: %.2f" % (self.id, self.e, self.n, self.u, self.h, self.p, self.r, self.g)
        return out

def mult_matrix_vec(m, v):
    """4x4 matrix transform of an XYZW coordinate - this matches OpenGL matrix conventions"""

    m = np.reshape(m, (4, 4)).T
    return np.matmul(m, v)
    
def get_bb_coords(client, i, screen_h, screen_w):
    """Calculates coordinates of intruder bounding box
    
    Parameters
    ----------
    client : SocketKind.SOCK_DGRAM
        XPlaneConnect socket
    i : int
        id number of ownship intruder instantiation
    screen_h, screen_w : int
        height and width of screen in pixels

    Returns
    -------
    int
        x position of intruder on screen from upper left 0,0
    int
        y position of intruder on screen from upper left 0,0
    """

    # retrieve x,y,z position of intruder
    acf_wrl = np.array([
        client.getDREF((f'sim/multiplayer/position/plane{i}_x'))[0],
        client.getDREF((f'sim/multiplayer/position/plane{i}_y'))[0],
        client.getDREF((f'sim/multiplayer/position/plane{i}_z'))[0],
        1.0
    ])
    
    mv = client.getDREF("sim/graphics/view/world_matrix")
    proj = client.getDREF("sim/graphics/view/projection_matrix_3d")
    
    acf_eye = mult_matrix_vec(mv, acf_wrl)
    acf_ndc = mult_matrix_vec(proj, acf_eye)
    
    acf_ndc[3] = 1.0 / acf_ndc[3]
    acf_ndc[0] *= acf_ndc[3]
    acf_ndc[1] *= acf_ndc[3]
    acf_ndc[2] *= acf_ndc[3]

    final_x = screen_w * (acf_ndc[0] * 0.5 + 0.5)
    final_y = screen_h * (acf_ndc[1] * 0.5 + 0.5)

    return final_x, screen_h - final_y

def set_position(client, aircraft):
    """Sets position of aircraft in XPlane

    Parameters
    ----------
    client : SocketKind.SOCK_DGRAM
        XPlaneConnect socket
    aircraft : Aircraft
        object containing details about craft's position
    """

    ref = [40.229635, -111.658833, 2000.0]
    p = pm.enu2geodetic(aircraft.e, aircraft.n, aircraft.u, ref[0], ref[1], ref[2]) #east, north, up
    client.sendPOSI([*p, aircraft.p, aircraft.r, aircraft.h, aircraft.g], aircraft.id)

def run_data_generation(client):
    """Begin data generation by calling gen_data"""

    client.pauseSim(True)
    
    aircraft_desc = client.getDREF("sim/aircraft/view/acf_descrip")
    byte_data = bytes(int(x) for x in aircraft_desc if x != 0)
    description = byte_data.decode('ascii')
    print("acf_descrip:", description)

    aircraft_icao_data = client.getDREF("sim/aircraft/view/acf_ICAO")
    byte_data = bytes(int(x) for x in aircraft_icao_data if x != 0)
    icao_code = byte_data.decode('ascii')
    print("ICAO code:", icao_code)

    # client.pauseSim(False)
    client.sendDREF("sim/operation/override/override_joystick", 1)
    # Set starting position of ownship and intruder
    set_position(client, Aircraft(1, -600, 1200, -10, 135, pitch=0, roll=0, gear=0))
    set_position(client, Aircraft(0, 0, 0, 0, 0, pitch=0, roll=0))
    client.sendDREFs([dome_offset_heading, dome_offset_pitch], [0, 0])
    client.sendVIEW(85)
    hwnd, abs_x, abs_y, width, height = wcw.get_xplane_window_info(window_title)
    # print(hwnd, abs_x, abs_y, width, height)
    screenshot = wcw.capture_xplane_window(hwnd, abs_x, abs_y, width, height)
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MPEG'), 30.0, (width, height))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    print(f"Screenshot shape: {screenshot.shape[1], screenshot.shape[0]}")
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(screenshot.shape[1]), int(screenshot.shape[0])))
    if not out.isOpened():
        print("Error: VideoWriter failed to open")
    # Pause to allow time for user to switch to XPlane window
    time.sleep(2)
    for i in range(300):
        set_position(client, Aircraft(1, -1000+i*5, 2000-i*5, 0, 135, pitch=0, roll=0))
        set_position(client, Aircraft(0, 0, i*3, 0, 0, pitch=0, roll=0))
        # if i == 0:
        time.sleep(0.1)
        if hwnd:
            screenshot = wcw.capture_xplane_window(hwnd, abs_x, abs_y, width, height)
            # print(f"Screenshot shape: {screenshot.shape}")
            bbc_x, bbc_y = get_bb_coords(client, 1, screenshot.shape[0], screenshot.shape[1])
            # print(f"Bounding box coordinates: {bbc_x}, {bbc_y}")
            cv2.circle(screenshot, (int(bbc_x), int(bbc_y)), 7, (0, 0, 255), 2)
            ret = out.write(screenshot)
            # print(ret)
            # if ret:
            #     print("Frame successfully written to video file")
            # else:
            #     print("Failed to write frame to video file")
            
            # try:
            #     ret = out.write(screenshot)
            #     if not ret:
            #         print("Error writing frame")
            # except cv2.error as e:
            #     print(f"OpenCV error: {e}")


            cv2.imshow('X-Plane Screenshot', screenshot)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cv2.imwrite(f'datasets\\test\\opencv_xplane_screenshot_{i}.png', screenshot)
            # print("Image shape: ", screenshot.shape)
            # print("Screenshot saved as 'opencv_xplane_screenshot.png'")
        else:
            print("X-Plane window not found.")
    
    out.release()
    cv2.destroyAllWindows()
    
        # time.sleep(0.033)
    # client.pauseSim(False)
    # for i in range(100):
    #     set_position(client, Aircraft(0, 0, i, 0, 0, pitch=0, roll=0))
    #     time.sleep(0.033)
    # for i in range(25):
    #     client.sendDREFs([dome_offset_pitch], [-i/2])
    #     time.sleep(0.033)
    #     current_pitch = client.getDREFs([dome_offset_pitch])
    #     # print(f"Current pitch: {current_pitch[0][0]}")
    # for i in range(360):
    #     # current_pitch = client.getDREFs(dome_offset_pitch)
    #     # print(f"Current pitch: {current_pitch}")
    #     client.sendDREFs([dome_offset_heading], [-i/2])
    #     time.sleep(0.033)
    #     # current_heading = client.getDREFs([dome_offset_heading, dome_offset_pitch]) 
    #     # print(f"Current Camera heading: {current_heading[1][0]} degrees")

with xpc.XPlaneConnect() as client:
    run_data_generation(client)