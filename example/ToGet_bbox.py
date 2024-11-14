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


window_title = 'X-System'
plot = True
camera_DREFs = ["sim/cockpit2/camera/camera_offset_heading",
        "sim/cockpit2/camera/camera_offset_pitch",
        "sim/cockpit2/camera/camera_offset_roll"]
dref_heading = "sim/cockpit2/camera/camera_offset_heading"
dref_pitch = "sim/cockpit2/camera/camera_offset_pitch"
dref_roll = "sim/cockpit2/camera/camera_offset_roll"
dref_view_roll = "sim/graphics/view/view_roll"
dome_offset_heading = "sim/graphics/view/dome_offset_heading"
dome_offset_pitch = "sim/graphics/view/dome_offset_pitch"
# ref = [40.669332, -74.012405, 1000.0] #new york
# ref = [24.979755, 121.451006, 500.0] #taiwan
ref = [40.229635, -111.658833, 20000.0] #provo
# ref = [-22.943736, -43.177820, 500.0] #rio
# ref = [38.870277, -77.030046, 500.0] #washington dc
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
    # retrieve x,y,z position of intruder
    acf_wrl = np.array([
        client.getDREF((f'sim/multiplayer/position/plane{i}_x'))[0],
        client.getDREF((f'sim/multiplayer/position/plane{i}_y'))[0],
        client.getDREF((f'sim/multiplayer/position/plane{i}_z'))[0],
        1.0
    ])
    print("acf_wrl:", acf_wrl)

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

def get_acf_poes_in_radians(i):
    # 獲取角度（假設角度為度數）
    yaw = client.getDREF(f'sim/multiplayer/position/plane{i}_psi')[0]
    pitch = client.getDREF(f'sim/multiplayer/position/plane{i}_the')[0]
    roll = client.getDREF(f'sim/multiplayer/position/plane{i}_phi')[0]
    # 將角度轉換為弧度
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    print(f"yaw: {yaw}, pitch: {pitch}, roll: {roll}")
    return yaw, -pitch, -roll

def get_rotation_matrix(yaw, pitch, roll):

    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    R = Ry @ Rx @ Rz
    return R

def get_bb_coords_by_icao(client, i, screen_h, screen_w):
    # retrieve x,y,z position of intruder
    acf_wrl = np.array([
        client.getDREF((f'sim/multiplayer/position/plane{i}_x'))[0],
        client.getDREF((f'sim/multiplayer/position/plane{i}_y'))[0],
        client.getDREF((f'sim/multiplayer/position/plane{i}_z'))[0],
        1.0
    ])
    R = get_rotation_matrix(*get_acf_poes_in_radians(i))
    nose_distance = 2.5
    nose_local = np.array([0, 0, nose_distance]) # nose is 2.5 meters in front of the cg (FRD)
    cg_world = acf_wrl[:3]
    nose_world = cg_world + R @ nose_local
    print("nose_world:", nose_world)

    aircraft_icao_data = client.getDREF("sim/aircraft/view/acf_ICAO")
    byte_data = bytes(int(x) for x in aircraft_icao_data if x != 0)
    icao_code = byte_data.decode('ascii')
    print("ICAO code:", icao_code)
    
    mv = client.getDREF("sim/graphics/view/world_matrix")
    proj = client.getDREF("sim/graphics/view/projection_matrix_3d")

    nose_wrl = np.append(nose_world, 1.0)
    diff = acf_wrl - nose_wrl
    print("diff:", diff)    
    acf_eye = mult_matrix_vec(mv, nose_wrl)
    acf_ndc = mult_matrix_vec(proj, acf_eye)
    
    acf_ndc[3] = 1.0 / acf_ndc[3]
    acf_ndc[0] *= acf_ndc[3]
    acf_ndc[1] *= acf_ndc[3]
    acf_ndc[2] *= acf_ndc[3]

    final_x = screen_w * (acf_ndc[0] * 0.5 + 0.5)
    final_y = screen_h * (acf_ndc[1] * 0.5 + 0.5)

    return final_x, screen_h - final_y



def projection_matrix_to_intrinsics(projection_matrix, width, height):
    # # fx, fy
    fx = projection_matrix[0, 0] * width / 2.0
    fy = projection_matrix[1, 1] * height / 2.0
    # # cx, cy
    cx = width * (1.0 + projection_matrix[0, 2]) / 2.0
    cy = height * (1.0 - projection_matrix[1, 2]) / 2.0

    return fx, fy, cx, cy

def set_position(client, aircraft, ref):
    """Set the position of the aircraft in the simulator"""
    p = pm.enu2geodetic(aircraft.e, aircraft.n, aircraft.u, ref[0], ref[1], ref[2]) #east, north, up
    client.sendPOSI([*p, aircraft.p, aircraft.r, aircraft.h, aircraft.g], aircraft.id)


# 定义一个全局变量来存储点击坐标
clicked_coords = None

def mouse_callback(event, x, y, flags, param):
    global clicked_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_coords = (x, y)
        print(f"Clicked at: {clicked_coords}")

def show_image(image):
    cv2.imshow('X-Plane Screenshot', image)
    cv2.setMouseCallback('X-Plane Screenshot', mouse_callback)  # 设置鼠标回调函数
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # 27 是ESC键的ASCII码
            break
    cv2.destroyAllWindows()

def pixel_to_world(coords, screen_w, screen_h):
    # 假设坐标转换函数：从图像空间到世界空间
    mv = client.getDREF("sim/graphics/view/world_matrix")
    proj = client.getDREF("sim/graphics/view/projection_matrix_3d")
    # projection_matrix_to_intrinsics(projection_matrix, screen_w, screen_h)

    u, v = coords
    x = (2.0 * u) / screen_w - 1.0
    y = 1.0 - (2.0 * v) / screen_h
    ndc = np.array([x, y, 1.0, 1.0])
    
    # 逆转换过程：从NDC到世界坐标 (这是一个简单示例，需要结合具体的数学)
    inv_proj = np.linalg.inv(proj)
    inv_mv = np.linalg.inv(mv)
    world_coords = inv_mv @ (inv_proj @ ndc)
    world_coords /= world_coords[3]  # 规范化

    return world_coords[:3]


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
    set_position(client, Aircraft(0, 0, 0, 0, 0, pitch=0, roll=0), ref)
    set_position(client, Aircraft(1, 0, 20, 0, 100, pitch=0, roll=0, gear=0), ref)
    # client.sendDREFs([dome_offset_heading, dome_offset_pitch], [0, 0])
    client.sendVIEW(85)
    time.sleep(1)
    if platform.system() == "Windows":
        hwnd, abs_x, abs_y, width, height = wcw.get_xplane_window_info(window_title)
        screenshot = wcw.capture_xplane_window(hwnd, abs_x, abs_y, width, height)
    else:
        xwininfo_output = subprocess.check_output(['xwininfo', '-name', 'X-System']).decode('utf-8')
        hwnd, abs_x, abs_y = so.get_xplane_window_info(xwininfo_output)
        screenshot = so.capture_xplane_window(hwnd, abs_x, abs_y)

    print(f"Screenshot shape: {screenshot.shape[1], screenshot.shape[0]}")

    ref_str = str(ref[0]) + "_" + str(ref[1]) + "_" + str(ref[2])
    print(f"Reference coordinates: {ref_str}")
    

    mv = client.getDREF("sim/graphics/view/world_matrix")
    proj = client.getDREF("sim/graphics/view/projection_matrix_3d")
    projection_matrix_3d = np.reshape(proj, (4, 4)).T
    
    fx, fy, cx, cy = projection_matrix_to_intrinsics(projection_matrix_3d, screenshot.shape[1], screenshot.shape[0])
    print(f"fx, fy, cx, cy: {fx, fy, cx, cy}")
    screenshot = screenshot.copy()
    bbc_x, bbc_y = get_bb_coords(client, 1, screenshot.shape[0], screenshot.shape[1])
    
    print(f"Bounding box coordinates: {bbc_x, bbc_y}")
    cv2.circle(screenshot, (int(bbc_x), int(bbc_y)), 1, (0, 0, 255), -1)

    nose_x, nose_y = get_bb_coords_by_icao(client, 1, screenshot.shape[0], screenshot.shape[1])
    print(f"Nose coordinates: {nose_x, nose_y}")
    cv2.circle(screenshot, (int(nose_x), int(nose_y)), 3, (0, 255, 0), -1)
    # Draw a cross at the center of the image
    center_x, center_y = screenshot.shape[1] // 2, screenshot.shape[0] // 2
    print(f"Center coordinates: {center_x, center_y}")
    cv2.line(screenshot, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 0), 1)
    cv2.line(screenshot, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 0), 1)
    cv2.imshow('X-Plane Screenshot', screenshot)


    # i = 1
    # acf_poes = np.array([
    #     client.getDREF((f'sim/multiplayer/position/plane{i}_psi'))[0],
    #     client.getDREF((f'sim/multiplayer/position/plane{i}_the'))[0],
    #     client.getDREF((f'sim/multiplayer/position/plane{i}_phi'))[0],
    #     1.0
    # ])
    # print(f"acf_poes: {acf_poes}")


    # cv2.setMouseCallback('X-Plane Screenshot', mouse_callback)  # 设置鼠标回调函数

    # zoom_factor = 2  # 放大倍數
    # zoom_size = 100  # 放大鏡的大小

    # while True:
    #     if clicked_coords:
    #         x, y = clicked_coords
    #         x1, y1 = max(0, x - zoom_size), max(0, y - zoom_size)
    #         x2, y2 = min(screenshot.shape[1], x + zoom_size), min(screenshot.shape[0], y + zoom_size)
    #         zoomed_region = screenshot[y1:y2, x1:x2]
    #         zoomed_region = cv2.resize(zoomed_region, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    #         zoomed_h, zoomed_w = zoomed_region.shape[:2]
    #         overlay = screenshot.copy()
    #         overlay[0:zoomed_h, 0:zoomed_w] = zoomed_region
    #         cv2.rectangle(overlay, (0, 0), (zoomed_w, zoomed_h), (255, 0, 0), 2)
    #         cv2.imshow('X-Plane Screenshot', overlay)
    #     else:
    #         cv2.imshow('X-Plane Screenshot', screenshot)

    #     if cv2.waitKey(1) & 0xFF == 27:
    #         break
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    
    if clicked_coords:
        world_coords = pixel_to_world(clicked_coords, screenshot.shape[1], screenshot.shape[0])
        print(f"World coordinates: {world_coords}")

    
with xpc.XPlaneConnect() as client:
    run_data_generation(client)