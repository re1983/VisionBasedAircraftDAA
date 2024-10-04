import pygetwindow as gw
import pyautogui
import numpy as np

# from PIL import Image

# def get_xplane_window_info():
#     window = gw.getWindowsWithTitle('X-System')
#     if window:
#         window = window[0]
#         return window
#     return None

# def capture_xplane_window(window):
#     window.activate()
#     screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
#     return screenshot

# window = get_xplane_window_info()

# if window:
#     screenshot = capture_xplane_window(window)
#     screenshot.save('xplane_screenshot.png')
#     print("Image shape: ", np.array(screenshot).shape)
#     print("Screenshot saved as 'xplane_screenshot.png'")
# else:
#     print("X-Plane window not found.")


# import ctypes
# from ctypes import wintypes

# # Constants for system metrics
# SM_CXFRAME = 32  # Width of the window border
# SM_CYFRAME = 33  # Height of the window border
# SM_CYCAPTION = 4  # Height of the caption area (title bar)

# # Load user32.dll
# user32 = ctypes.WinDLL('user32', use_last_error=True)

# def get_window_border_and_titlebar_size():
#     # Get border sizes
#     border_width = user32.GetSystemMetrics(SM_CXFRAME)
#     border_height = user32.GetSystemMetrics(SM_CYFRAME)
#     titlebar_height = user32.GetSystemMetrics(SM_CYCAPTION)

#     return border_width, border_height, titlebar_height

# # Example usage
# border_width, border_height, titlebar_height = get_window_border_and_titlebar_size()
# print(f"Border Width: {border_width}px")
# print(f"Border Height: {border_height}px")
# print(f"Title Bar Height: {titlebar_height}px")

import cv2

def get_xplane_window_info(window_title):
    window = gw.getWindowsWithTitle(window_title)
    if window:
        window = window[0]
        return window._hWnd, window.left, window.top, window.width, window.height
    return None, None, None, None, None

def capture_xplane_window(hwnd, abs_x, abs_y, width, height):
    # Adjust the region to exclude the window borders and title bar
    border_pixels = 8  # Adjust this value based on your window border size
    title_bar_pixels = 31  # Adjust this value based on your window title bar size
    screenshot = pyautogui.screenshot(region=(abs_x + border_pixels, abs_y + title_bar_pixels, width - 2 * border_pixels, height - title_bar_pixels - border_pixels))
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# window_title = 'X-System'
# hwnd, abs_x, abs_y, width, height = get_xplane_window_info(window_title)
# print("Window handle: ", hwnd)
# print("Window position: ", abs_x, abs_y)
# print("Window size: ", width, height)

# if hwnd:
#     screenshot = capture_xplane_window(hwnd, abs_x, abs_y, width, height)
#     cv2.imwrite('datasets\\test\\opencv_xplane_screenshot.png', screenshot)
#     print("Image shape: ", screenshot.shape)
#     print("Screenshot saved as 'opencv_xplane_screenshot.png'")
# else:
#     print("X-Plane window not found.")



# from fast_ctypes_screenshots import ScreenshotOfWindow
# import win32gui

# def capture_specific_window(window_title):
#     hwnd = win32gui.FindWindow(None, window_title)
    
#     if hwnd:
#         with ScreenshotOfWindow(hwnd=hwnd, client=False, ascontiguousarray=False) as screenshot:
#             img = screenshot.screenshot_window()
#             # 這裡的img是一個numpy數組,你可以使用PIL或OpenCV來保存它
#             img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR if needed
#             cv2.imwrite(f'{window_title}_screenshot.png', img)
#         print(f"Screenshot of {window_title} captured and saved as '{window_title}_screenshot.png'")
#     else:
#         print("Window not found")

# # 使用示例
# capture_specific_window('X-System')

# import win32gui
# import win32api
# import mss

# def capture_window(window_title):
#     # 獲取視窗句柄
#     hwnd = win32gui.FindWindow(None, window_title)
    
#     if hwnd:
#         # 獲取視窗位置和大小
#         rect = win32gui.GetWindowRect(hwnd)
#         x, y, width, height = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
        
#         # 使用MSS擷取該區域
#         with mss.mss() as sct:
#             monitor = {"top": y, "left": x, "width": width, "height": height}
#             screenshot = sct.grab(monitor)
            
#             # 保存截圖
#             mss.tools.to_png(screenshot.rgb, screenshot.size, output=f"{window_title}.png")
        
#         print(f"Screenshot saved as {window_title}.png")
#     else:
#         print("Window not found")

# # 使用示例
# capture_window("X-System")