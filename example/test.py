import sys
import xpc
import PID
from datetime import datetime, timedelta
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.Qt import QtWidgets
import time
import pymap3d as pm

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
    
    def __init__(self, ac_num, east, north, up, heading, pitch=-998, roll=-998):
        self.id = ac_num
        self.e = east
        self.n = north
        self.u = up
        self.h = heading
        self.p = pitch
        self.r = roll
    
    def __str__(self):
        out = "Craft: %.2f, East: %.2f, North: %.2f, Up: %.2f, Heading: %.2f, Pitch: %.2f, Roll: %.2f" % (self.id, self.e, self.n, self.u, self.h, self.p, self.r)
        return out

def set_position(client, aircraft):
    """Sets position of aircraft in XPlane

    Parameters
    ----------
    client : SocketKind.SOCK_DGRAM
        XPlaneConnect socket
    aircraft : Aircraft
        object containing details about craft's position
    """

    ref = [42.37415695190430, -71.01775360107422, 500.0]
    p = pm.enu2geodetic(aircraft.e, aircraft.n, aircraft.u, ref[0], ref[1], ref[2]) #east, north, up
    client.sendPOSI([*p, aircraft.p, aircraft.r, aircraft.h], aircraft.id)

def run_data_generation(client):
    """Begin data generation by calling gen_data"""

    client.pauseSim(True)
    client.sendDREF("sim/operation/override/override_joystick", 1)
    # client.sendDREF("sim/cockpit2/switches/camera_power_on", 1)
    # Set starting position of ownship and intruder
    set_position(client, Aircraft(1, 0, 100, 0, 0, pitch=0, roll=0))
    set_position(client, Aircraft(0, 0, 0, 0, 0, pitch=0, roll=0))
    client.sendDREFs([dome_offset_heading, dome_offset_pitch], [0, 0])
    client.sendVIEW(85)
    # client.sendVIEW(xpc.ViewType.Tower)
    # dref_camera_type = "sim/cockpit2/camera/camera_type"
    # client.sendDREFs([dref_camera_type], [80])  # Track view (跟蹤視角)
    # print("Camera type set to Track view (following the aircraft)")
    # dref_camera_type = "sim/cockpit2/camera/camera_type"
    # client.sendDREFs([dref_camera_type], [3])
    # current_dref_camera_type = client.getDREFs([dref_camera_type])
    # print(f"Camera type : ({current_dref_camera_type[0]})")

    # Pause to allow time for user to switch to XPlane window
    # time.sleep(1)
    for i in range(100):
        set_position(client, Aircraft(1, 0, 100+i, 0, 135, pitch=0, roll=i))
        time.sleep(0.033)
    # client.pauseSim(False)
    for i in range(100):
        set_position(client, Aircraft(0, 0, i, 0, 0, pitch=0, roll=0))
        time.sleep(0.033)
    for i in range(100):
        client.sendDREFs([dome_offset_heading, dome_offset_pitch], [i, -i])
        time.sleep(0.033)
        # current_heading = client.getDREFs([dome_offset_heading, dome_offset_pitch]) 
        # print(f"Current Camera heading: {current_heading[0]} degrees")

with xpc.XPlaneConnect() as client:
    run_data_generation(client)