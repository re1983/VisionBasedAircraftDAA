import time
import xpc

# 連接到X-Plane
client = xpc.XPlaneConnect()

# 設定player aircraft的初始位置和方向 (latitude, longitude, altitude, pitch, roll, heading)
player_initial_pos = [37.524, -122.06899, 5000, 0, 0, 0]  # 正北飛行
client.sendPOSI(player_initial_pos, 0)

# 設定non-player aircraft的初始位置和方向 (latitude, longitude, altitude, pitch, roll, heading)
non_player_initial_pos = [37.524, -122.07899, 5000, 0, 0, 90]  # 由西向東飛行
client.sendPOSI(non_player_initial_pos, 1)

# 設定飛行速度 (airspeed in knots)
player_speed = 250
non_player_speed = 250
client.sendDREF("sim/flightmodel/position/indicated_airspeed", player_speed)
# client.sendDREF("sim/flightmodel/position/indicated_airspeed", non_player_speed, 1)

# # 設定non-player aircraft的機型 (例如: "B738" 代表波音737-800)
# non_player_aircraft_type = "B738"
# client.sendDREF("sim/multiplayer/position/plane1_ICAO", non_player_aircraft_type)

# 模擬飛行
try:
    for _ in range(100):  # 模擬100個時間步
        # 更新player aircraft的位置 (往正北飛)()
        player_initial_pos[0] += 0.0001  # 更新緯度
        client.sendPOSI(player_initial_pos, 0)

        # 更新non-player aircraft的位置 (由西向東飛)
        non_player_initial_pos[1] += 0.0001  # 更新經度
        client.sendPOSI(non_player_initial_pos, 1)

        # 等待0.1秒
        time.sleep(0.1)
finally:
    # 斷開連接
    client.close()