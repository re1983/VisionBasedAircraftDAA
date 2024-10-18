import pycolmap
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from pycolmap import logging


project_directory = Path('project_directory')
image_dir = project_directory / 'test'  
database_path = project_directory / 'database.db'  
sparse_output_path = project_directory / 'sparse' 
input_path = project_directory / 'input'


# 创建一个新的 COLMAP 重建对象
reconstruction = pycolmap.Reconstruction()
# 手动载入相机的内参和外参
reconstruction.read_text(str(sparse_output_path))

print(reconstruction.cameras)
print(reconstruction.images)
# # # 打印重建信息
print(reconstruction.summary())
# 执行增量式 SfM
reconstructions = pycolmap.triangulate_points(
    reconstruction,
    str(database_path), 
    str(image_dir), 
    str(sparse_output_path),
)
reconstructions.write_text(str(sparse_output_path))