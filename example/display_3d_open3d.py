import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import pycolmap
import numpy as np


project_directory = Path('project_directory')
image_dir = project_directory / 'test'  # 图片存储的文件夹
database_path = project_directory / 'database.db'  # 创建一个数据库文件
sparse_output_path = project_directory / 'sparse'  # 存储稀疏结果的文件夹
input_path = project_directory / 'input'  # 包含 images.txt 和 cameras.txt 的文件夹

import open3d as o3d
import numpy as np
import struct

# 读取 points3D.bin 文件
def read_points3D_binary(path_to_model_file):
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = struct.unpack('<Q', fid.read(8))[0]
        for _ in range(num_points):
            binary_point_line_properties = fid.read(43)
            point3D_id, x, y, z, r, g, b, error = struct.unpack('<Q3d3Bd', binary_point_line_properties)
            points3D[point3D_id] = np.array([x, y, z, r, g, b])
    return points3D

# 读取数据
points3D = read_points3D_binary(sparse_output_path /'points3D.bin')

# 创建点云对象
pcd = o3d.geometry.PointCloud()
points = np.array([point[:3] for point in points3D.values()])
colors = np.array([point[3:6] for point in points3D.values()]) / 255.0

pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 可视化
o3d.visualization.draw_geometries([pcd])
