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
ref = [40.0, -111.658833, 5000.0] #provo
# ref = [-22.943736, -43.177820, 500.0] #rio
# ref = [38.870277, -77.030046, 500.0] #washington dc
name_list_points = ["Nose", "Tail", "Right", "Left", "Top", "Bottom"]
color_list_points = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
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

# def get_rotation_matrix(yaw, pitch, roll):

#     # 将角度转换为弧度

#     Rz = np.array([
#         [np.cos(roll), -np.sin(roll), 0],
#         [np.sin(roll), np.cos(roll), 0],
#         [0, 0, 1]
#     ])

#     Ry = np.array([
#         [np.cos(yaw), 0, np.sin(yaw)],
#         [0, 1, 0],
#         [-np.sin(yaw), 0, np.cos(yaw)]
#     ])

#     Rx = np.array([
#         [1, 0, 0],
#         [0, np.cos(pitch), -np.sin(pitch)],
#         [0, np.sin(pitch), np.cos(pitch)]
#     ])

#     R = Rz @ Ry @ Rx
#     print("R = ", R)
#     return R

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

def get_the_ponits(icao_code):
    
    if  icao_code == "C172":
        points = np.array([
            [0, -0.05, -2.5],   #Nose of the aircraft
            [0, 1.4, 5.82],   #Tail of the aircraft
            [5.5, 0.7, 0],    #Right wing tip
            [-5.5, 0.7, 0],   #Left wing tip
            [0, 1.52, 5.2],   #Top of the aircraft
            [0, -1.37, 0]      #Bottom of the aircraft
        ])
    elif icao_code == "B738":
        points = np.array([
            [0, 0, -3.5],
            [0, 0, 10.5],
            [10.5, 0, 0],
            [-10.5, 0, 0],
            [0, 2, 0],
            [0, -1, 0]
        ])
    

    return points

def generate_bounding_box(nose, tail, right, left, top, bottom):

    center_z = (nose[2] + tail[2]) / 2
    center_x = (right[0] + left[0]) / 2
    center_y = (top[1] + bottom[1]) / 2

    half_length = np.linalg.norm(np.array(nose) - np.array(tail)) / 2
    half_width = np.linalg.norm(np.array(right) - np.array(left)) / 2
    half_height = np.linalg.norm(np.array(top) - np.array(bottom)) / 2

    vertices = np.array([
        [center_x + half_width, center_y + half_height, center_z + half_length],  # 前右上
        [center_x + half_width, center_y - half_height, center_z + half_length],  # 前右下
        [center_x - half_width, center_y + half_height, center_z + half_length],  # 前左上
        [center_x - half_width, center_y - half_height, center_z + half_length],  # 前左下
        [center_x + half_width, center_y + half_height, center_z - half_length],  # 后右上
        [center_x + half_width, center_y - half_height, center_z - half_length],  # 后右下
        [center_x - half_width, center_y + half_height, center_z - half_length],  # 后左上
        [center_x - half_width, center_y - half_height, center_z - half_length]   # 后左下 
    ])
    print("vertices:\n", vertices)
    return vertices

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


def get_bb_coords_by_icao(client, i, screen_h, screen_w):

    aircraft_icao_data = client.getDREF("sim/aircraft/view/acf_ICAO")
    byte_data = bytes(int(x) for x in aircraft_icao_data if x != 0)
    icao_code = byte_data.decode('ascii')
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
    # nose_distance = 2.5
    # nose_local = np.array([0, 0, nose_distance]) # nose is 2.5 meters in front of the cg (FRD)
    cg_world = acf_wrl[:3]

    points = get_the_ponits(icao_code)
    nose = points[0]
    tail = points[1]
    right = points[2]
    left = points[3]
    top = points[4]
    bottom = points[5]
    vertices = generate_bounding_box_GC(nose, tail, right, left, top, bottom)
    # print("vertices:", vertices)
    points_world = np.array([cg_world + R @ point for point in points])
    
    # print("points_world:", points_world)
    # nose_world = cg_world + R @ nose_local
    # nose_world = R @ cg_world + nose_local

    # print("nose_world:", nose_world)

    # nose_wrl = np.append(nose_world, 1.0)
    # diff = acf_wrl - nose_wrl
    # print("diff:", diff)

    # acf_eye = mult_matrix_vec(mv, nose_wrl)
    # acf_ndc = mult_matrix_vec(proj, acf_eye)
    
    # Nose_x, Nose_y = get_projection_xy(points_world[0], acf_wrl, mv, proj, screen_h, screen_w)
    list_points_xy = get_projections_xy(points_world, acf_wrl, mv, proj, screen_h, screen_w)
    vertices_world = np.array([cg_world + R @ point for point in vertices])
    list_vertices_xy = get_projections_xy(vertices_world, acf_wrl, mv, proj, screen_h, screen_w) 
    # print("list_points_xy:", list_points_xy)
    # Tail_x, Tail_y = get_projection_xy(client, i, screen_h, screen_w)

    # return Nose_x, Nose_y
    return list_points_xy, list_vertices_xy
    # acf_ndc[3] = 1.0 / acf_ndc[3]
    # acf_ndc[0] *= acf_ndc[3]
    # acf_ndc[1] *= acf_ndc[3]
    # acf_ndc[2] *= acf_ndc[3]

    # final_x = screen_w * (acf_ndc[0] * 0.5 + 0.5)
    # final_y = screen_h * (acf_ndc[1] * 0.5 + 0.5)nose

    # return final_x, screen_h - final_y



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
    

def run_data_generation(client):
    """Begin data generation by calling gen_data"""
    # client.pauseSim(False)
    # Set starting position of ownship and intruder
    set_position(client, Aircraft(0, 0, 0, 0, heading=0, pitch=0, roll=0), ref)
    set_position(client, Aircraft(1, 0, 30, 0, heading=45, pitch=45, roll=45, gear=0), ref)
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


    points_list, vertices_list = get_bb_coords_by_icao(client, 1, screenshot.shape[0], screenshot.shape[1])
    for i, point in enumerate(points_list):
        cv2.circle(screenshot, (int(point[0]), int(point[1])), 3, color_list_points[i], 1)
        cv2.putText(screenshot, name_list_points[i], (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_list_points[i], 1)
    # Draw Convex Hull bounding box for points_list
    hull = cv2.convexHull(np.array(points_list, dtype=np.int32))
    cv2.polylines(screenshot, [hull], isClosed=True, color=(0, 255, 0), thickness=2)
    x, y, w, h = cv2.boundingRect(hull)
    cv2.rectangle(screenshot, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for vertex in vertices_list:
        cv2.circle(screenshot, (int(vertex[0]), int(vertex[1])), 3, (0, 0, 255), -1)
    # cv2.circle(screenshot, (int(nose_x), int(nose_y)), 3, (0, 255, 0), -1)
    
    # Draw a cross at the center of the image
    Draw_a_cross_at_the_center_of_the_image(screenshot)
    cv2.imshow('X-Plane Screenshot', screenshot)

    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    
    if clicked_coords:
        world_coords = pixel_to_world(clicked_coords, screenshot.shape[1], screenshot.shape[0])
        print(f"World coordinates: {world_coords}")


with xpc.XPlaneConnect() as client:
    client.pauseSim(True)
    client.sendDREF("sim/operation/override/override_joystick", 1)
    run_data_generation(client)