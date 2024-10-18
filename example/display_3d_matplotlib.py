import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设 read_write_model.py 位于同一目录下
import read_write_model as colmap_utils

# 读取 points3D.bin 文件
points3D = colmap_utils.read_points3D_binary('project_directory/sparse/points3D.bin') 
# project_directory/sparse/0/points3D.bin
# print(f"Total points: {len(points3D)}")
# print(points3D)
cameras = colmap_utils.read_cameras_binary("project_directory/sparse/cameras.bin")
# print(f"Total points: {len(cameras)}")
# print(cameras)


# 提取 X, Y, Z 坐标
x_vals = []
y_vals = []
z_vals = []

for point_id, point_data in points3D.items():
    x_vals.append(point_data.xyz[0])
    y_vals.append(point_data.xyz[1])
    z_vals.append(point_data.xyz[2])

# 使用 Matplotlib 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_vals, y_vals, z_vals, s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置 XYZ 轴等比
ax.set_box_aspect([1, 1, 1])

plt.show()
