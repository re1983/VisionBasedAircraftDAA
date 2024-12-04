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
ref = [40.0, -111.658833, 3500.0] #provo
# ref = [-22.943736, -43.177820, 500.0] #rio
# ref = [38.870277, -77.030046, 500.0] #washington dc
name_list_points = ["Nose", "Tail", "Right", "Left", "Top", "Bottom"]
color_list_points = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
class Aircraft:
    """Object for storing positional information for Aircraft"""
    def __init__(self, ac_num, east, north, up, heading=-998, pitch=-998, roll=-998, gear=-998):
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
    # print("acf_wrl:", acf_wrl)

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
    print(f"yaw: {yaw}, pitch: {pitch}, roll: {roll} in radians")
    return yaw, pitch, roll

def get_acf_poes_in_euler(i):
    # 獲取角度（假設角度為度數）
    yaw = client.getDREF(f'sim/multiplayer/position/plane{i}_psi')[0]
    pitch = client.getDREF(f'sim/multiplayer/position/plane{i}_the')[0]
    roll = client.getDREF(f'sim/multiplayer/position/plane{i}_phi')[0]
    print(f"yaw: {yaw}, pitch: {pitch}, roll: {roll} in degrees")
    return yaw, pitch, roll

from scipy.spatial.transform import Rotation as R

def euler_to_quaternion(yaw, pitch, roll):
    # 将角度转换为弧度
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    # 使用 scipy.spatial.transform.Rotation 进行转换
    rotation = R.from_euler('xyz', [roll, pitch, yaw])
    quaternion = rotation.as_quat()  # 返回 (x, y, z, w)
    # X-Plane uses East-Up-South (EUS) coordinates, convert to North-East-Down (NED)
    # q_ned = np.array([quaternion[3], quaternion[0], -quaternion[2], -quaternion[1]])
    # print(f"Quaternion in NED: {q_ned}")
    return quaternion

def quaternion_to_rotation_matrix(quaternion):
    # 使用 scipy.spatial.transform.Rotation 进行转换
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    return rotation_matrix

def get_rotation_matrix_quaternion_radians(yaw, pitch, roll):
    quaternion = euler_to_quaternion(yaw, pitch, roll)
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    print("Rotation Matrix: \n", rotation_matrix)
    return rotation_matrix

def get_rotation_matrix(yaw, pitch, roll):

    rotation = R.from_euler('zxy', [-roll, pitch, -yaw], degrees=True)
    rotation_matrix = rotation.as_matrix()
    return rotation_matrix

def get_projection_xy(point_world, acf_wrl, mv, proj, screen_h, screen_w):
    
    point_wrl = np.append(point_world, 1.0)
    diff = acf_wrl - point_wrl
    print("acf_wrl - point_wrl = ", diff)    
    acf_eye = mult_matrix_vec(mv, point_wrl)
    acf_ndc = mult_matrix_vec(proj, acf_eye)
    
    acf_ndc[3] = 1.0 / acf_ndc[3]
    acf_ndc[0] *= acf_ndc[3]
    acf_ndc[1] *= acf_ndc[3]
    acf_ndc[2] *= acf_ndc[3]

    final_x = screen_w * (acf_ndc[0] * 0.5 + 0.5)
    final_y = screen_h * (acf_ndc[1] * 0.5 + 0.5)

    return final_x, screen_h - final_y

def get_projections_xy(points_world, acf_wrl, mv, proj, screen_h, screen_w):
    projected_points = []
    for point_world in points_world:
        point_wrl = np.append(point_world, 1.0)
        acf_eye = mult_matrix_vec(mv, point_wrl)
        acf_ndc = mult_matrix_vec(proj, acf_eye)

        acf_ndc[3] = 1.0 / acf_ndc[3]
        acf_ndc[0] *= acf_ndc[3]
        acf_ndc[1] *= acf_ndc[3]
        acf_ndc[2] *= acf_ndc[3]

        final_x = screen_w * (acf_ndc[0] * 0.5 + 0.5)
        final_y = screen_h * (acf_ndc[1] * 0.5 + 0.5)

        projected_points.append((final_x, screen_h - final_y))
    return projected_points

def get_the_geometry_ponits(icao_code):
    
    if  icao_code == "C172": # Cessna Skyhawk https://en.wikipedia.org/wiki/Cessna_172
        points = np.array([
            [0, -0.05, -2.5],   #Nose of the aircraft
            [0, 0, 5.82],   #Tail of the aircraft
            [5.5, 0.7, 0],    #Right wing tip
            [-5.5, 0.7, 0],   #Left wing tip
            [0, 1.52, 5.82],   #Top of the aircraft
            [0, -1.37, 0]      #Bottom of the aircraft
        ])
        cruise_speed = 122.0 # 122 kn (140 mph, 226 km/h)
        ADG_group = 1 # Airplane Design Group
        # done

    elif icao_code == "BE58": # Beechcraft Baron 58 https://en.wikipedia.org/wiki/Beechcraft_Baron
        points = np.array([
            [0, -0.3, -3],   #Nose of the aircraft
            [0, 0.15, 5.85],   #Tail of the aircraft
            [5.4, -0.1, 0.4],    #Right wing tip
            [-5.4, -0.1, 0.4],   #Left wing tip
            [0, 1.9, 5.85],   #Top of the aircraft
            [0, -1.2, 0]      #Bottom of the aircraft
        ])
        cruise_speed = 180.0 # 180 kn (210 mph, 330 km/h)
        ADG_group = 1 # Airplane Design Group
        # done

    elif icao_code == "BE9L": # Beechcraft King Air C90 https://en.wikipedia.org/wiki/Beechcraft_King_Air
        points = np.array([
            [0, 0.4, -3.6],   #Nose of the aircraft
            [0, 0.75, 7.3],   #Tail of the aircraft
            [7.9, 1, 0.5],    #Right wing tip
            [-7.9, 1, 0.5],   #Left wing tip
            [0, 3.4, 7.3],   #Top of the aircraft
            [0, -1.0, -1.5]      #Bottom of the aircraft
        ]) # done
        cruise_speed = 226.0 # 226 kn (260 mph, 426 km/h)
        ADG_group = 2 # Airplane Design Group
        # done

    elif icao_code == "B738": # Boeing 737-800 https://en.wikipedia.org/wiki/Boeing_737
        points = np.array([
            [0, 0.1, -18.5],   #Nose of the aircraft
            [0, 0, 22],   #Tail of the aircraft
            [18, 3, 7],    #Right wing tip
            [-18, 3, 7],   #Left wing tip
            [0, 9.7, 21],   #Top of the aircraft
            [0, -2.85, -0]      #Bottom of the aircraft
        ])
        cruise_speed = 453.0 # Mach 0.785 (453 kn; 838 km/h; 521 mph)
        ADG_group = 3 # Airplane Design Group

    elif icao_code == "SF50": # Cirrus Vision SF50 https://en.wikipedia.org/wiki/Cirrus_Vision_SF50
        points = np.array([
            [0, -0.2, -3.7],   #Nose of the aircraft
            [0, 0.0, 5.82],   #Tail of the aircraft
            [5.9, -0.2, 1],    #Right wing tip
            [-5.9, -0.2, 1],   #Left wing tip
            [0, 1.7, 5.82],   #Top of the aircraft
            [0, -1.5, 0]      #Bottom of the aircraft
        ])
        cruise_speed = 305.0 # 305 kn (351 mph, 565 km/h)
        ADG_group = 1 # Airplane Design Group
        # done

    elif icao_code == "S76": # Sikorsky S-76C https://en.wikipedia.org/wiki/Sikorsky_S-76
        points = np.array([
            [0, 1.0, -6.7],   #Nose of the aircraft
            [0, -0.5, 9.3],   #Tail of the aircraft
            [6.7, 1, 0],    #Right wing tip
            [-6.7, 1, 0],   #Left wing tip
            [0, 2.7, 9.3],   #Top of the aircraft
            [0, -1.8, -3.5]      #Bottom of the aircraft
        ])
        cruise_speed =  155.0 # 155 kn (178 mph, 287 km/h)
        ADG_group = 1 # Airplane Design Group
        # done
    
    return points, cruise_speed, ADG_group

def generate_bounding_box_GC(nose, tail, right, left, top, bottom):
    # 将所有点放入一个数组
    points = np.array([nose, tail, right, left, top, bottom])

    # 计算最小和最大坐标
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    # 计算中心点和半边长
    center = (min_coords + max_coords) / 2
    half_lengths = (max_coords - min_coords) / 2

    # 生成顶点
    vertices = np.array([
        [center[0] + half_lengths[0], center[1] + half_lengths[1], center[2] + half_lengths[2]],  # 前右上
        [center[0] + half_lengths[0], center[1] - half_lengths[1], center[2] + half_lengths[2]],  # 前右下
        [center[0] - half_lengths[0], center[1] + half_lengths[1], center[2] + half_lengths[2]],  # 前左上
        [center[0] - half_lengths[0], center[1] - half_lengths[1], center[2] + half_lengths[2]],  # 前左下
        [center[0] + half_lengths[0], center[1] + half_lengths[1], center[2] - half_lengths[2]],  # 后右上
        [center[0] + half_lengths[0], center[1] - half_lengths[1], center[2] - half_lengths[2]],  # 后右下
        [center[0] - half_lengths[0], center[1] + half_lengths[1], center[2] - half_lengths[2]],  # 后左上
        [center[0] - half_lengths[0], center[1] - half_lengths[1], center[2] - half_lengths[2]]   # 后左下 
    ])
    print("vertices:\n", vertices)
    return vertices

def get_list_acfs_icao(client):
    icao_code_acf_list = []
    icao_types = client.getDREF("sim/cockpit2/tcas/targets/icao_type")
    # print("TCAS ICAO Types:", icao_types)
    icao_codes = [icao_types[i:i+8] for i in range(0, len(icao_types), 8)]
    for i, code in enumerate(icao_codes):
        icao_code = ''.join([chr(int(x)) for x in code if x != 0])
        # print(f"Aircraft {i} ICAO code: {icao_code}")
        if icao_code != '':
            icao_code_acf_list.append(icao_code)

    return icao_code_acf_list

def get_bb_coords_by_icao(client, i, screen_h, screen_w):

    # aircraft_icao_data = client.getDREF("sim/aircraft/view/acf_ICAO")
    # byte_data = bytes(int(x) for x in aircraft_icao_data if x != 0)
    # icao_code = byte_data.decode('ascii')
    # print("ICAO code:", icao_code)

    # retrieve x,y,z position of intruder
    acf_wrl = np.array([
        client.getDREF((f'sim/multiplayer/position/plane{i}_x'))[0],
        client.getDREF((f'sim/multiplayer/position/plane{i}_y'))[0],
        client.getDREF((f'sim/multiplayer/position/plane{i}_z'))[0],
        1.0
    ])
    mv = client.getDREF("sim/graphics/view/world_matrix")
    proj = client.getDREF("sim/graphics/view/projection_matrix_3d")
    R = get_rotation_matrix(*get_acf_poes_in_euler(i))
    cg_world = acf_wrl[:3]
    # points = get_the_ponits(icao_code)
    icao_code_acf_list = get_acf_icao(client)
    print("ICAO code list:", icao_code_acf_list)
    print("ICAO code:", icao_code_acf_list[i])
    points, cruise_speed, ADG_group = get_the_geometry_ponits(icao_code_acf_list[i])
    nose = points[0]
    tail = points[1]
    right = points[2]
    left = points[3]
    top = points[4]
    bottom = points[5]
    vertices = generate_bounding_box_GC(nose, tail, right, left, top, bottom)
    points_world = np.array([cg_world + R @ point for point in points])
    list_points_xy = get_projections_xy(points_world, acf_wrl, mv, proj, screen_h, screen_w)
    vertices_world = np.array([cg_world + R @ point for point in vertices])
    list_vertices_xy = get_projections_xy(vertices_world, acf_wrl, mv, proj, screen_h, screen_w) 

    return list_points_xy, list_vertices_xy

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


def Draw_a_cross_at_the_center_of_the_image(screenshot):
    center_x, center_y = screenshot.shape[1] // 2, screenshot.shape[0] // 2
    print(f"Image Center coordinates: {center_x, center_y}")
    cv2.line(screenshot, (center_x - 10, center_y), (center_x + 10, center_y), (0, 200, 0), 1)
    cv2.line(screenshot, (center_x, center_y - 10), (center_x, center_y + 10), (0, 200, 0), 1)

def Draw_Convex_Hull_bounding_box_for_six_points(screenshot, points_list):
    hull = cv2.convexHull(np.array(points_list, dtype=np.int32))
    cv2.polylines(screenshot, [hull], isClosed=True, color=(0, 255, 0), thickness=1)
    x, y, w, h = cv2.boundingRect(hull)
    cv2.rectangle(screenshot, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # Oriented Bounding Boxes Object https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    OBB = cv2.minAreaRect(hull)
    obb_box = cv2.boxPoints(OBB)
    obb_box = np.int0(obb_box)
    cv2.drawContours(screenshot, [obb_box], 0, (0, 0, 255), 1)

def Draw_bounding_cube_for_eigth_corners_vertices(screenshot, vertices_list):
    for vertex in vertices_list:
        cv2.circle(screenshot, (int(vertex[0]), int(vertex[1])), 1, (0, 0, 255), -1)

def run_data_generation(client):
    """Begin data generation by calling gen_data"""
    # client.pauseSim(False)
    # Set starting position of ownship and intruder
    set_position(client, Aircraft(0, 0, 0, 0, heading=0, pitch=0, roll=0), ref)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    set_position(client, Aircraft(1, 0, 1000, 0, heading=0, pitch=0, roll=0, gear=0), ref)
    # client.sendDREFs([dome_offset_heading, dome_offset_pitch], [0, 0])
    client.sendVIEW(85)
    time.sleep(0.03)
    if platform.system() == "Windows":
        hwnd, abs_x, abs_y, width, height = wcw.get_xplane_window_info(window_title)
        screenshot = wcw.capture_xplane_window(hwnd, abs_x, abs_y, width, height)
    else:
        xwininfo_output = subprocess.check_output(['xwininfo', '-name', 'X-System']).decode('utf-8')
        hwnd, abs_x, abs_y = so.get_xplane_window_info(xwininfo_output)
        screenshot = so.capture_xplane_window(hwnd, abs_x, abs_y)

    print(f"Screenshot shape: {screenshot.shape[1], screenshot.shape[0]}")
    print(f"Reference coordinates: {ref}")
    
    mv = client.getDREF("sim/graphics/view/world_matrix")
    proj = client.getDREF("sim/graphics/view/projection_matrix_3d")
    projection_matrix_3d = np.reshape(proj, (4, 4)).T
    
    fx, fy, cx, cy = projection_matrix_to_intrinsics(projection_matrix_3d, screenshot.shape[1], screenshot.shape[0])
    print(f"fx, fy, cx, cy: {fx, fy, cx, cy}")
    screenshot = screenshot.copy()
    bbc_x, bbc_y = get_bb_coords(client, 1, screenshot.shape[0], screenshot.shape[1])
    
    print(f"Bounding box coordinates: {bbc_x, bbc_y}")
    cv2.circle(screenshot, (int(bbc_x), int(bbc_y)), 1, (0, 0, 255), -1)

    points_list, vertices_list = get_bb_coords_by_icao(client, 1, screenshot.shape[0], screenshot.shape[1])

    for i, point in enumerate(points_list):
        cv2.circle(screenshot, (int(point[0]), int(point[1])), 3, color_list_points[i], 1)
        cv2.putText(screenshot, name_list_points[i], (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_list_points[i], 1)
    
    # Draw_bounding_cube_for_eigth_corners_vertices(screenshot, vertices_list)
    Draw_Convex_Hull_bounding_box_for_six_points(screenshot, points_list)
    Draw_a_cross_at_the_center_of_the_image(screenshot)

    cv2.imshow('X-Plane Screenshot', screenshot)

    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    
    if clicked_coords:
        world_coords = pixel_to_world(clicked_coords, screenshot.shape[1], screenshot.shape[0])
        print(f"World coordinates: {world_coords}")


def generate_positions_by_timestep(start_point, heading, fps, duration,velocity_kn):
    """Generate a list of positions based on starting point, heading, fps, and velocity in knots."""
    positions = []
    velocity_mps = velocity_kn * 0.5144444444  # Convert knots to meters per second
    time_step = 1 / fps  # Time step in seconds

    current_position = np.array(start_point)
    heading = (90 - heading) % 360
    heading_rad = np.radians(heading)

    for _ in range(duration*fps):
        delta_x = velocity_mps * time_step * np.cos(heading_rad)
        delta_y = velocity_mps * time_step * np.sin(heading_rad)
        current_position += np.array([delta_x, delta_y])
        positions.append(current_position.copy())

    return positions


import matplotlib.pyplot as plt
import random_path_gen as rpg

def plot_positions(positions):
    """Plot a list of positions on a 2D plane."""
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Aircraft Positions')
    plt.grid(True)
    plt.show()

# FOV = 60
# near1, far1 = 50, 1000
# near2, far2 = 50, 1000 
# offset1 = (0, 0)
# offset2 = (0, 550)
# heading, point1, point2 = rpg.get_random_points_between_two_trapezoids(FOV, near1, far1, near2, far2, offset1, offset2)
# print(f"heading: {heading}, point1: {point1}, point2: {point2}")
# positions_list = generate_positions(point1, heading, 30, 10, 155)
# print(f"Number of positions: {len(positions_list)}")
# plot_positions(positions_list)

def get_bb_coords_acfs(client, Number_of_aircrafts, proj_mat, screen_h, screen_w):
    bb_coords = []
    mv = client.getDREF("sim/graphics/view/world_matrix")
    # proj = client.getDREF("sim/graphics/view/projection_matrix_3d")
    # retrieve x,y,z position of intruder
    for i in range(Number_of_aircrafts):
        if i == 0:
            continue
        else:
            acf_wrl = np.array([
                client.getDREF((f'sim/multiplayer/position/plane{i}_x'))[0],
                client.getDREF((f'sim/multiplayer/position/plane{i}_y'))[0],
                client.getDREF((f'sim/multiplayer/position/plane{i}_z'))[0],
                1.0
            ])

            acf_eye = mult_matrix_vec(mv, acf_wrl)
            acf_ndc = mult_matrix_vec(proj_mat, acf_eye)
        
            acf_ndc[3] = 1.0 / acf_ndc[3]
            acf_ndc[0] *= acf_ndc[3]
            acf_ndc[1] *= acf_ndc[3]
            acf_ndc[2] *= acf_ndc[3]

            final_x = screen_w * (acf_ndc[0] * 0.5 + 0.5)
            final_y = screen_h * (acf_ndc[1] * 0.5 + 0.5)
            bb_coords.append((final_x, screen_h - final_y))

    return bb_coords


def run_data_generation_sequentially(client):
    all_positions_in_path = []
    fps = 30
    duration = 10

    client.sendVIEW(85)
    if platform.system() == "Windows":
        hwnd, abs_x, abs_y, width, height = wcw.get_xplane_window_info(window_title)
        screenshot = wcw.capture_xplane_window(hwnd, abs_x, abs_y, width, height)
    else:
        xwininfo_output = subprocess.check_output(['xwininfo', '-name', 'X-System']).decode('utf-8')
        hwnd, abs_x, abs_y = so.get_xplane_window_info(xwininfo_output)
        screenshot = so.capture_xplane_window(hwnd, abs_x, abs_y)

    print(f"Screenshot shape: {screenshot.shape[1], screenshot.shape[0]}")
    print(f"Reference coordinates: {ref}")
   
    mv = client.getDREF("sim/graphics/view/world_matrix")
    proj = client.getDREF("sim/graphics/view/projection_matrix_3d")
    projection_matrix_3d = np.reshape(proj, (4, 4)).T
    
    fx, fy, cx, cy = projection_matrix_to_intrinsics(projection_matrix_3d, screenshot.shape[1], screenshot.shape[0])
    print(f"fx, fy, cx, cy: {fx, fy, cx, cy}")

    """Begin data generation by calling gen_data"""
    X_FOV = client.getDREF("sim/graphics/view/field_of_view_deg")[0]
    print(f"Field of View: {X_FOV}")
    near1, far1 = 50, 1600
    near2, far2 = 50, 1600 

    icao_code_acf_list = get_list_acfs_icao(client)
    print("ICAO code list:", icao_code_acf_list)
    print("Number of aircrafts:", len(icao_code_acf_list))
    # Set starting position of ownship and intruder
    for i, icao_code in enumerate(icao_code_acf_list):

        if i == 0:
            set_position(client, Aircraft(i, 0, 0, 0, heading=0, pitch=0, roll=0), ref) # ownship
            points, cruise_speed, ADG_group = get_the_geometry_ponits(icao_code)
            path = generate_positions_by_timestep([0.0,0.0], 0, fps, duration, cruise_speed)
            # print(path)
            all_positions_in_path.append(path)
            offset1 = path[0]
            # print(f"offset1: {offset1}")
            offset2 = path[-1]
            # print(f"offset2: {offset2}")
        else:
            client.sendDREF(f"sim/multiplayer/position/plane{i}_throttle", 0.0)
            points, cruise_speed, ADG_group = get_the_geometry_ponits(icao_code)
            heading, point1, point2 = rpg.get_random_points_between_two_trapezoids(X_FOV, near1, far1, near2, far2, offset1, offset2)
            # print(f"heading: {heading}, point1: {point1}, point2: {point2}")
            set_position(client, Aircraft(i, point1[0], point1[1], -30, heading=heading, pitch=0, roll=0), ref)
            path = generate_positions_by_timestep(point1, heading, fps, duration, cruise_speed)
            all_positions_in_path.append(path)

    time.sleep(3)
    # np.save('all_positions_in_path.npy', all_positions_in_path)


    for t in range(len(all_positions_in_path[0])):

        # print(f"Time step: {t}")
        # time.sleep(1/fps*2)

        for j, positions in enumerate(all_positions_in_path):
            # print(f"Setting position for aircraft {j} at {positions[t]}")
            if j == 0:
                set_position(client, Aircraft(j, positions[t][0], positions[t][1], 0, heading=-998, pitch=-998, roll=-998), ref)
            else:
                set_position(client, Aircraft(j, positions[t][0], positions[t][1], -30, heading=-998, pitch=-998, roll=-998), ref)

        time.sleep(1/fps)

        if hwnd:
            if platform.system() == "Windows":
                screenshot = wcw.capture_xplane_window(hwnd, abs_x, abs_y, width, height)
            else:
                screenshot = so.capture_xplane_window(hwnd, abs_x, abs_y)  

        bbc_list = get_bb_coords_acfs(client, len(icao_code_acf_list), proj, screenshot.shape[0], screenshot.shape[1])
        screenshot = screenshot.copy()
        for i, bbc in enumerate(bbc_list):
            cv2.circle(screenshot, (int(bbc[0]), int(bbc[1])), 1, (0, 0, 255), -1)
        
        cv2.imshow('X-Plane Screenshot', screenshot)
        # screenshot = screenshot.copy()

        if cv2.waitKey(1) & 0xFF == 27:
            break
    

    cv2.destroyAllWindows()
    # for path in all_positions_in_path[0]:
    # for t in range(len(all_positions_in_path[0])):
    #     for j, pos in enumerate(positions):
    #         print(f"Setting position for aircraft {j} at {pos}")

    # for i, positions in enumerate(all_positions_in_path):
    #     print(f"Setting position for aircraft {i} at {path}")
        # set_position(client, Aircraft(i, positions[0], positions[1], -998, heading=-998, pitch=-998, roll=-998), ref)

        # nose = points[0]
        # tail = points[1]
        # right = points[2]
        # left = points[3]
        # top = points[4]
        # bottom = points[5]
        # vertices = generate_bounding_box_GC(nose, tail, right, left, top, bottom)
        # print(f"ICAO code: {icao_code}")
        # print(f"points: {points}")
        # print(f"vertices: {vertices}")
        # set_position(client, Aircraft(i, 0, 0, 0, heading=0, pitch=0, roll=0), ref)
        # time.sleep(0.1)
        # set_position(client, Aircraft(i+1, 0, 1000, 0, heading=0, pitch=0, roll=0, gear=0), ref)
        # time.sleep(0.1)
        # list_points_xy, list_vertices_xy = get_bb_coords_by_icao(client, i+1, 1080, 1920)
        # print(f"list_points_xy: {list_points_xy}")
        # print(f"list_vertices_xy: {list_vertices_xy}")
        # screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Draw_Convex_Hull_bounding_box_for_six_points(screenshot, list_points_xy)
        # Draw_bounding_cube_for_eigth_corners_vertices(screenshot, list_vertices_xy)
        # Draw_a_cross_at_the_center_of_the_image(screenshot)
        # cv2.imshow('X-Plane Screenshot', screenshot)
        # while True:
        #     if cv2.waitKey(1) & 0xFF == 27:
        #         break
        # cv2.destroyAllWindows()

    # set_position(client, Aircraft(1, point1[0], point1[1], 0, heading=heading, pitch=0, roll=0, gear=0), ref)

    # heading, point1, point2 = rpg.get_random_points_between_two_trapezoids(FOV, near1, far1, near2, far2, offset1, offset2)
    # print(f"heading: {heading}, point1: {point1}, point2: {point2}")
    # set_position(client, Aircraft(2, point1[0], point1[1], 0, heading=0, pitch=0, roll=0, gear=0), ref)

    # heading, point1, point2 = rpg.get_random_points_between_two_trapezoids(FOV, near1, far1, near2, far2, offset1, offset2)
    # print(f"heading: {heading}, point1: {point1}, point2: {point2}")
    # set_position(client, Aircraft(3, point1[0], point1[1], 0, heading=0, pitch=0, roll=0, gear=0), ref)

    # heading, point1, point2 = rpg.get_random_points_between_two_trapezoids(FOV, near1, far1, near2, far2, offset1, offset2)
    # print(f"heading: {heading}, point1: {point1}, point2: {point2}")
    # set_position(client, Aircraft(4, point1[0], point1[1], 0, heading=0, pitch=0, roll=0, gear=0), ref)

    # heading, point1, point2 = rpg.get_random_points_between_two_trapezoids(FOV, near1, far1, near2, far2, offset1, offset2)
    # print(f"heading: {heading}, point1: {point1}, point2: {point2}")
    # set_position(client, Aircraft(5, point1[0], point1[1], 0, heading=0, pitch=0, roll=0, gear=0), ref)




with xpc.XPlaneConnect() as client:
    client.pauseSim(False)
    # time.sleep(0.5)
    client.pauseSim(True)
    client.sendDREF("sim/operation/override/override_joystick", 1)
    # run_data_generation(client)
    run_data_generation_sequentially(client)