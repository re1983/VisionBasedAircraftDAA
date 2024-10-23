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
import platform
# Conditional import based on OS
if platform.system() == "Windows":
    import win_capturewindow as wcw
else:
    import subprocess
    import screenshot_Opencv as so

write_oupt_flag = True  
plot = False

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

def projection_matrix_to_intrinsics(projection_matrix, width, height):
    # # fx, fy
    fx = projection_matrix[0, 0] * width / 2.0
    fy = projection_matrix[1, 1] * height / 2.0
    # # cx, cy
    cx = width * (1.0 + projection_matrix[0, 2]) / 2.0
    cy = height * (1.0 - projection_matrix[1, 2]) / 2.0

    return fx, fy, cx, cy

def write_cameras_txt(camera_id, fx, fy, cx, cy, width, height, filename="project_directory/input/cameras.txt"):
    with open(filename, 'w') as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

def write_images_txt(image_data, filename="project_directory/input/images.txt"):
    with open(filename, 'w') as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for image_id, (qw, qx, qy, qz, tx, ty, tz, image_name) in enumerate(image_data, start=1):
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {image_name}\n")
            f.write("\n")

def matrix_to_quaternion(R):
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]
    QW = np.sqrt(1.0 + r11 + r22 + r33) / 2.0
    QX = (r32 - r23) / (4.0 * QW)
    QY = (r13 - r31) / (4.0 * QW)
    QZ = (r21 - r12) / (4.0 * QW)

    return QW, QX, QY, QZ

def extract_rotation_translation(world_matrix):
    R = world_matrix[:3, :3]
    t = world_matrix[:3, 3]
    QW, QX, QY, QZ = matrix_to_quaternion(R)
    
    return QW, QX, QY, QZ, t


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
    # sim/graphics/view/acf_matrix
    # sim/graphics/view/window_height
    # sim/graphics/view/window_width
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

def set_position(client, aircraft, ref):
    """Sets position of aircraft in XPlane

    Parameters
    ----------
    client : SocketKind.SOCK_DGRAM
        XPlaneConnect socket
    aircraft : Aircraft
        object containing details about craft's position
    """
    p = pm.enu2geodetic(aircraft.e, aircraft.n, aircraft.u, ref[0], ref[1], ref[2]) #east, north, up
    client.sendPOSI([*p, aircraft.p, aircraft.r, aircraft.h, aircraft.g], aircraft.id)

ref = [40.757, -73.973, 1300.0] #new york
# ref = [24.979755, 121.451006, 500.0] #taiwan
# ref = [40.2291, -111.6587, 2000.0] #provo

# ref = [40.244319, -111.649226, 1600.0] #byu
# ref = [-22.943736, -43.177820, 500.0] #rio
# ref = [38.870277, -77.030046, 500.0] #washington dc
# ref = [40.216836, -111.717362, 1450.0] #provo airport

def run_data_generation(client):
    """Begin data generation by calling gen_data"""

    client.sendDREF("sim/operation/override/override_joystick", 1)
    client.pauseSim(True)
    
    aircraft_desc = client.getDREF("sim/aircraft/view/acf_descrip")
    byte_data = bytes(int(x) for x in aircraft_desc if x != 0)
    description = byte_data.decode('ascii')
    print("acf_descrip:", description)

    aircraft_icao_data = client.getDREF("sim/aircraft/view/acf_ICAO")
    byte_data = bytes(int(x) for x in aircraft_icao_data if x != 0)
    icao_code = byte_data.decode('ascii')
    print("ICAO code:", icao_code)
    # Set starting position of ownship and intruder
    set_position(client, Aircraft(0, 0, 0, 0, 0, pitch=0, roll=0), ref)
    client.sendDREFs([dome_offset_heading, dome_offset_pitch], [0, -90])
    client.sendVIEW(85)

    # mv = client.getDREF("sim/graphics/view/world_matrix")
    # print(f"World matrix: {mv}")
    # print(f"World matrix length: {len(mv)}")

    # client.pauseSim(False)

    if platform.system() == "Windows":
        hwnd, abs_x, abs_y, width, height = wcw.get_xplane_window_info(window_title)
        # print(hwnd, abs_x, abs_y, width, height)
        screenshot = wcw.capture_xplane_window(hwnd, abs_x, abs_y, width, height)
    else:
        xwininfo_output = subprocess.check_output(['xwininfo', '-name', 'X-System']).decode('utf-8')
        hwnd, abs_x, abs_y = so.get_xplane_window_info(xwininfo_output)
        screenshot = so.capture_xplane_window(hwnd, abs_x, abs_y)

    proj = client.getDREF("sim/graphics/view/projection_matrix_3d")
    # print(f"Projection matrix: {proj}")
    projection_matrix_3d = np.reshape(proj, (4, 4)).T
    print('Projection matrix:')
    print(projection_matrix_3d)
    np.save('project_directory/input/projection_matrix_3d.npy', projection_matrix_3d)
    fx, fy, cx, cy = projection_matrix_to_intrinsics(projection_matrix_3d, screenshot.shape[1], screenshot.shape[0])
    if write_oupt_flag:
        write_cameras_txt(camera_id=1, fx=fx, fy=fy, cx=cx, cy=cy, width=screenshot.shape[1], height=screenshot.shape[0])
    print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}, width: {screenshot.shape[1]}, height: {screenshot.shape[0]}")

    ref_str = str(ref[0]) + "_" + str(ref[1]) + "_" + str(ref[2])
    print(f"Reference coordinates: {ref_str}")
    
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # out = cv2.VideoWriter((ref_str + '_output.mp4'), fourcc, 30.0, (int(screenshot.shape[1]), int(screenshot.shape[0])))
    # if not out.isOpened():
    #     print("Error: VideoWriter failed to open")
    # Pause to allow time for user to switch to XPlane window
    time.sleep(1)

    for i in range(240):
        set_position(client, Aircraft(0, 0, i*10, 0, 0, pitch=0, roll=0), ref)
        time.sleep(0.1)
        if hwnd:
            if platform.system() == "Windows":
                screenshot = wcw.capture_xplane_window(hwnd, abs_x, abs_y, width, height)
            else:
                screenshot = so.capture_xplane_window(hwnd, abs_x, abs_y)      
            # print(f"Screenshot shape: {screenshot.shape}")
            if plot == True:
                screenshot = screenshot.copy()
                # bbc_x, bbc_y = get_bb_coords(client, 1, screenshot.shape[0], screenshot.shape[1])
                # cv2.circle(screenshot, (int(bbc_x), int(bbc_y)), 3, (0, 0, 255), -1)
                # bbc_x, bbc_y = get_bb_coords(client, 2, screenshot.shape[0], screenshot.shape[1])
                # cv2.circle(screenshot, (int(bbc_x), int(bbc_y)), 3, (0, 255, 0), -1)

            # ret = out.write(screenshot)
            # cv2.imshow('X-Plane Screenshot', screenshot)
            if write_oupt_flag:
                cv2.imwrite(f'project_directory/test/{i:04d}.png', screenshot) #datasets\test

            # print("Image shape: ", screenshot.shape, "i: ", i)
            wv = client.getDREF("sim/graphics/view/world_matrix")
            # print(f"World matrix: {wv}")
            
            wv4_4 = np.reshape(wv, (4, 4)).T
            if write_oupt_flag:
                np.save(f'project_directory/test/{i:04d}.npy', wv4_4)
                print(f"World matrix: {wv4_4}")
            # print(f"World matrix length: {len(wv4_4)}")
            # print(wv4_4)
            quaternion_translation = extract_rotation_translation(wv4_4)
            # print(f"Quaternion and translation: {quaternion_translation}")
            image_id=i, 
            qw=quaternion_translation[0]
            qx=quaternion_translation[1]
            qy=quaternion_translation[2]
            qz=quaternion_translation[3] 
            tx=quaternion_translation[4][0]
            ty=quaternion_translation[4][1]
            tz=quaternion_translation[4][2]
            image_name=(f'{i:04d}.png')
            image_data.append((qw, qx, qy, qz, tx, ty, tz, image_name))
            # print(f"Image data: {image_data[-1]}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("X-Plane window not found.")
    
    # out.release()
    print("Data generation complete.")
    # print("Image data: ", image_data)
    if write_oupt_flag:
        write_images_txt(image_data)
    cv2.destroyAllWindows()
    



image_data = []
with xpc.XPlaneConnect() as client:
    run_data_generation(client)