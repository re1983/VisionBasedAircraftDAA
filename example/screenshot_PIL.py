from Xlib import display, X
from PIL import Image
import numpy as np
import subprocess

def get_xplane_window_info(xwininfo_output):
    window_id = None
    abs_x = None
    abs_y = None

    for line in xwininfo_output.splitlines():

        if "Window id" in line:
            window_id = int(line.split()[3], 16)
        if "Absolute upper-left X" in line:
            abs_x = int(line.split()[-1])
        if "Absolute upper-left Y" in line:
            abs_y = int(line.split()[-1])

    return window_id, abs_x, abs_y


def capture_xplane_window(window_id, abs_x, abs_y):
    dsp = display.Display()
    root = dsp.screen().root
    window = dsp.create_resource_object('window', window_id)
    geom = window.get_geometry()
    raw = root.get_image(abs_x, abs_y, geom.width, geom.height, X.ZPixmap, 0xffffffff)    
    img = Image.frombytes("RGB", (geom.width, geom.height), raw.data, "raw", "BGRX")
    return img


xwininfo_output = subprocess.check_output(['xwininfo', '-name', 'X-System']).decode('utf-8')
window_id, abs_x, abs_y = get_xplane_window_info(xwininfo_output)


if window_id:
    screenshot = capture_xplane_window(window_id, abs_x, abs_y)
    screenshot.save('xplane_screenshot.png')
    print("Image shape: ", np.array(screenshot).shape)
    print("Screenshot saved as 'xplane_screenshot.png'")
else:
    print("X-Plane window not found.")
