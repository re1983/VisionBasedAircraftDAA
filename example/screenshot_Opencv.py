from Xlib import display, X
import numpy as np
import subprocess
import cv2

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
    # Convert raw data to numpy array
    img = np.frombuffer(raw.data, dtype=np.uint8).reshape((geom.height, geom.width, 4))
    # Convert from BGRX to BGR
    img = img[:, :, :3]
    
    return img


xwininfo_output = subprocess.check_output(['xwininfo', '-name', 'X-System']).decode('utf-8')
window_id, abs_x, abs_y = get_xplane_window_info(xwininfo_output)

if window_id:
    screenshot = capture_xplane_window(window_id, abs_x, abs_y)
    cv2.imwrite('opencv_xplane_screenshot.png', screenshot)
    print("Image shape: ", screenshot.shape)
    print("Screenshot saved as 'xplane_screenshot.png'")
else:
    print("X-Plane window not found.")
