import pycolmap
import os
from pathlib import Path

# 设定数据的目录
project_directory = Path('project_directory')
image_path = project_directory / 'test'  # 图片存储的文件夹
database_path = project_directory / 'database.db'  # 创建一个数据库文件
sparse_output_path = project_directory / 'sparse'  # 存储稀疏结果的文件夹
input_path = project_directory / 'input'  # 包含 images.txt 和 cameras.txt 的文件夹

# 创建输出目录
# sparse_output_path.mkdir(parents=True, exist_ok=True)

# 创建一个新的 COLMAP 重建对象
# reconstruction = pycolmap.Reconstruction()

# # # 手动载入相机的内参和外参
# reconstruction.read_text(str(input_path))

# # # 打印重建信息
# print(reconstruction.summary())

# 特征提取
pycolmap.extract_features(str(database_path), str(image_path))

# 特征匹配
pycolmap.match_exhaustive(str(database_path))

num_images = pycolmap.Database(database_path).num_images
print(f"共有 {num_images} 张图片")
print(pycolmap.summary())

# 执行增量式 SfM
reconstructions = pycolmap.incremental_mapping(
    str(database_path), 
    str(image_path), 
    str(sparse_output_path),
    input_path=str(input_path)  # 使用已有的相机和图像信息
)

# 保存重建结果
for i, rec in reconstructions.items():
    rec.write_text(str(sparse_output_path / f'reconstruction_{i}'))


print("重建完成，结果保存在:", sparse_output_path)