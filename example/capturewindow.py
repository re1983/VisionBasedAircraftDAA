from Xlib import display, X
from PIL import Image
import numpy as np
import subprocess

# 获取 X-Plane 窗口 ID
def get_xplane_window():
    # 使用 `xwininfo` 获取窗口信息
    xwininfo_output = subprocess.check_output(['xwininfo', '-name', 'X-System']).decode('utf-8')
    # print(xwininfo_output)
    # 解析窗口 ID
    for line in xwininfo_output.splitlines():
        if "Window id" in line:
            # print(line.split()[3], 16)
            return int(line.split()[3], 16)

def get_xplane_window_info(xwininfo_output):
    window_id = None
    abs_x = None
    abs_y = None

    for line in xwininfo_output.splitlines():
        # 提取 Window id
        if "Window id" in line:
            window_id = int(line.split()[3], 16)
        # 提取 Absolute upper-left X 和 Absolute upper-left Y
        if "Absolute upper-left X" in line:
            abs_x = int(line.split()[-1])
        if "Absolute upper-left Y" in line:
            abs_y = int(line.split()[-1])

    # 返回 Window id 和绝对坐标
    return window_id, abs_x, abs_y

# 截图
def capture_xplane_window(window_id, abs_x, abs_y):
    dsp = display.Display()
    root = dsp.screen().root
    window = dsp.create_resource_object('window', window_id)
    geom = window.get_geometry()
    # print(geom)
    # 轉換坐標為絕對坐標
    # coords = window.translate_coords(root, geom.x, geom.y)

    # x_root, y_root = root.translate_coords(window, 0, 0)
    # print(x_root, y_root)
    # print(coords.x, coords.y)
    raw = root.get_image(abs_x, abs_y, geom.width, geom.height, X.ZPixmap, 0xffffffff)
    # raw = root.get_image(geom.x, geom.y, geom.width, geom.height, X.ZPixmap, 0xffffffff)
    
    img = Image.frombytes("RGB", (geom.width, geom.height), raw.data, "raw", "BGRX")
    return img

# 获取窗口 ID
xwininfo_output = subprocess.check_output(['xwininfo', '-name', 'X-System']).decode('utf-8')
window_id, abs_x, abs_y = get_xplane_window_info(xwininfo_output)

# 截取并保存 X-Plane 窗口截图
if window_id:
    screenshot = capture_xplane_window(window_id, abs_x, abs_y)
    screenshot.save('xplane_screenshot.png')
    print("Screenshot saved as 'xplane_screenshot.png'")
else:
    print("X-Plane window not found.")
