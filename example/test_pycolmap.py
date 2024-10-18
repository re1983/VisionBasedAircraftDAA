# https://github.com/colmap/pycolmap/blob/master/example.py
# https://colmap.github.io/index.html

import pycolmap
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from pycolmap import logging

# 设定数据的目录
project_directory = Path('project_directory')
image_dir = project_directory / 'test'  # 图片存储的文件夹
database_path = project_directory / 'database.db'  # 创建一个数据库文件
sparse_output_path = project_directory / 'sparse'  # 存储稀疏结果的文件夹
input_path = project_directory / 'input'  # 包含 images.txt 和 cameras.txt 的文件夹

# 创建输出目录
# sparse_output_path.mkdir(parents=True, exist_ok=True)



# # 读取 cameras.txt 文件
# with open(input_path / 'cameras.txt', 'r') as f:
#     content = f.read()

# # 解析 cameras.txt 内容
# camera_params = []
# for line in content.strip().split('\n'):
#     if line and not line.startswith('#'):
#         parts = line.split()
#         camera_params = parts

# # 提取相机参数
# camera_id = int(camera_params[0])
# model = camera_params[1]
# width = int(camera_params[2])
# height = int(camera_params[3])
# focal_length_x = float(camera_params[4])
# focal_length_y = float(camera_params[5])
# principal_point_x = float(camera_params[6])
# principal_point_y = float(camera_params[7])

# # 创建 pycolmap.Camera 对象
# camera = pycolmap.Camera(
#     model=model,
#     width=width,
#     height=height,
#     params=[focal_length_x, focal_length_y, principal_point_x, principal_point_y]
# )

# print(f"Camera ID: {camera_id}")
# print(f"Camera Model: {camera.model}")
# print(f"Image Size: {camera.width}x{camera.height}")
# print(f"Focal Length: {camera.focal_length_x}, {camera.focal_length_y}")
# print(f"Principal Point: ({camera.principal_point_x}, {camera.principal_point_y})")

# reconstruction.add_camera(camera)
# # print(f"Camera ID: {camera_id}")
# # print(f"Reconstruction object: {reconstruction}")
# print(f"Cameras in reconstruction: {reconstruction.cameras}")

# # # # 打印重建信息
# print(reconstruction.summary())


# # 从 images.txt 加载图像信息
# with open(input_path / 'images.txt', 'r') as f:
#     lines = f.readlines()

# for line in lines:
#     if line.startswith('#') or len(line.strip()) == 0:
#         continue
#     parts = line.split()
#     print(parts)
#     if len(parts) >= 10:
#         image_id = int(parts[0])
#         qvec = [float(q) for q in parts[1:5]]
#         rotation_matrix = R.from_quat(qvec).as_matrix()
#         tvec = [float(t) for t in parts[5:8]]
#         image_name = parts[9]
#         image = pycolmap.Image(
#             image_id=image_id,
#             cam_from_world = pycolmap.Rigid3d(rotation=rotation_matrix, translation=tvec),
#             # qvec=qvec,
#             # tvec=tvec,
#             camera_id=camera_id,
#             name=image_name
#         )
#         reconstruction.add_image(image)
# print(f"Images in reconstruction: {reconstruction.images}")


# 手动载入相机的内参和外参
# reconstruction.read_text(str(input_path))

# print(reconstruction.cameras)
# print(reconstruction.images)
# # # 打印重建信息
# print(reconstruction.summary())

# image_list = sorted(os.listdir(image_dir))
# image_list = [os.path.join(image_dir, file)
#                for file in os.listdir(image_dir)
#                if file.endswith(('.jpg', '.png'))]
# image_list.sort()  # 按字母或數字順序排列
# print(image_list)


# # # 特征提取
# for image in image_list:
#     image_path = os.path.join(image_dir, image)
#     print(image_path)
#     # pycolmap.extract_features(database_path, image_path)
#     pycolmap.extract_features(database_path = str(database_path), image_path = str(image_dir), camera_mode = "SINGLE"
#                             , camera_model='PINHOLE')
# pycolmap.extract_features(reconstruction,  camera_mode = "SINGLE", camera_model='PINHOLE')

# 自定義 SIFT 特徵提取的參數
sift_options = pycolmap.SiftExtractionOptions()
sift_options.num_threads = 1  # 設置為單線程

pycolmap.extract_features(database_path = str(database_path), image_path = str(image_dir), camera_mode = "SINGLE"
                            , camera_model='PINHOLE', sift_options=sift_options)
            #  camera_params=[focal_length_x, focal_length_y, principal_point_x, principal_point_y])

# # 特征匹配
# pycolmap.match_exhaustive(str(database_path))
pycolmap.match_sequential(str(database_path))
# database.commit()

# db = pycolmap.Database(str(database_path))


num_images = pycolmap.Database(database_path).num_images
print(f"共有 {num_images} 张图片")
# print("重建完成，结果保存在:", sparse_output_path)

# import enlighten

# with enlighten.Manager() as manager:
#     with manager.counter(total=num_images, desc="Images registered:") as pbar:
#         pbar.update(0, force=True)
#         recs = pycolmap.incremental_mapping(
#         str(database_path), 
#         str(image_dir), 
#         str(sparse_output_path),
#         input_path=str(input_path),  # 使用已有的相机和图像信息
#             initial_image_pair_callback=lambda: pbar.update(2),
#             next_image_callback=lambda: pbar.update(1),
#         )
# logging.set_log_destination(logging.INFO, sparse_output_path / "INFO.log.")  # + time
# for idx, rec in recs.items():
#     logging.info(f"#{idx} {rec.summary()}")


# 创建一个新的 COLMAP 重建对象
reconstruction = pycolmap.Reconstruction()
# 手动载入相机的内参和外参
reconstruction.read_text(str(input_path))

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

# 保存重建结果
# for i, rec in reconstructions.items():
#     reconstruction_dir = sparse_output_path / f'reconstruction_{i}'
#     reconstruction_dir.mkdir(parents=True, exist_ok=True)
#     rec.write_text(str(reconstruction_dir))