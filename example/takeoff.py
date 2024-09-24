# from XPlaneConnect import XPlaneConnect
import xpc
import time

# 創建 XPlaneConnect 客戶端
client = xpc.XPlaneConnect()

# 獲取當前經緯度
latitude_dref = 'sim/flightmodel/position/latitude'
longitude_dref = 'sim/flightmodel/position/longitude'

# 獲取經緯度值
latitude = client.getDREFs([latitude_dref])[0][0]  # 獲取緯度
longitude = client.getDREFs([longitude_dref])[0][0]  # 獲取經度

# 確保飛機在跑道上
drefs = [
    'sim/flightmodel/position/groundspeed',
    latitude_dref,
    longitude_dref
]
values = [0, latitude, longitude]  # 替換成實際的經緯度

client.sendDREFs(drefs, values)

# 啟動引擎
engine_dref = 'sim/engine/ENGN_thro'
client.sendDREFs([engine_dref], [1.0])  # 全油門，確保這裡的 DREFs 和 values 數量相同
# time.sleep(5)

# # 開始滑行
# # client.sendDREFs('sim/flightmodel/position/groundspeed', [10])  # 設定地速

# # 當達到起飛速度時，拉起飛機
# time.sleep(10)  # 根據你的飛機特性調整這個時間
# client.sendDREFs('sim/flightmodel/position/phi', [15])  # 上升角度
