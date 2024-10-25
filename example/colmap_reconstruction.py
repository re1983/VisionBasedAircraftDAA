import os
import pycolmap
import sqlite3
import numpy as np
import cv2


def projection_matrix_to_intrinsics(projection_matrix, width, height):
    """
    从 X-Plane/OpenGL 的 projection_matrix_3d 中提取内参矩阵，结合图像分辨率 (width, height)
    """
    fx = projection_matrix[0, 0] * width / 2.0
    fy = projection_matrix[1, 1] * height / 2.0
    cx = width * (1.0 + projection_matrix[0, 2]) / 2.0
    cy = height * (1.0 - projection_matrix[1, 2]) / 2.0

    return fx, fy, cx, cy

def opengl_to_colmap(world_matrix):
    # OpenGL 的 world_matrix 是 4x4 矩阵
    rotation_matrix = world_matrix[:3, :3]
    translation_vector = world_matrix[:3, 3]
    return rotation_matrix, translation_vector

# 数据集路径
image_dir = 'project_directory/test/'  # 图像文件夹路径
database_path = 'project_directory/colmap.db'  # COLMAP 数据库
output_dir = 'project_directory/output/'  # 输出文件夹

# Initialize reconstruction
reconstruction = pycolmap.Reconstruction()

# # 假设你有300张图像
# num_images = 300

# 手动添加相机模型信息
projection_matrix_3d = np.load('project_directory/input/projection_matrix_3d.npy')
# 讀出的一張圖的寬高
sample_image_path = next(
    os.path.join(image_dir, f) for f in os.listdir(image_dir)
    if f.endswith('.png') or f.endswith('.jpg')
)
print(sample_image_path)
sample_image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
height, width = sample_image.shape[1], sample_image.shape[0]
# 假设使用 pinhole 模型，fx, fy, cx, cy 分别为从 X-Plane 的 projection_matrix_3d 计算出来的值
fx, fy, cx, cy = projection_matrix_to_intrinsics(projection_matrix_3d, width, height)
camera_params = [fx, fy, cx, cy]
# Add camera (assuming PINHOLE model)
camera_id = reconstruction.add_camera(
    model="PINHOLE",  # You can use other models depending on your camera
    width=width,
    height=height,
    params=camera_params
)

for filename in sorted(os.listdir(image_dir)):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        img_path = os.path.join(image_dir, filename)

        # Load corresponding world_matrix for the current image
        matrix_path = os.path.join(image_dir, filename.replace('.png', '.npy').replace('.jpg', '.npy'))
        world_matrix = np.load(matrix_path)

        # Convert world_matrix to COLMAP-compatible rotation and translation
        rotation_matrix, translation_vector = opengl_to_colmap(world_matrix)

        # Add image with pose to reconstruction
        reconstruction.add_image(
            name=filename, 
            camera_id=camera_id, 
            qvec=pycolmap.RotationMatrix(rotation_matrix).to_quaternion(),  # Convert rotation matrix to quaternion
            tvec=translation_vector
        )

# Optionally, save the reconstruction to disk
reconstruction.write('project_directory/output')

# Optionally, view the reconstruction details
print(reconstruction)

# camera_id = db.add_camera(
#     model='PINHOLE',
#     width=width,
#     height=height,
#     params=[fx, fy, cx, cy]
# )

# # 打开数据库
# conn = sqlite3.connect(database_path)
# cursor = conn.cursor()

# CAMERA_MODEL = 1  # PINHOLE camera model
# # 插入相机信息到 cameras 表
# cursor.execute('''
#     INSERT INTO cameras (model, width, height, params, prior_focal_length) 
#     VALUES (?, ?, ?, ?, ?)
# ''', (CAMERA_MODEL, width, height, np.array(camera_params, dtype=np.float64).tobytes(), 0))  # 1 for PINHOLE camera model, 0 for prior_focal_length
# camera_id = cursor.lastrowid

# for filename in sorted(os.listdir(image_dir)):
#     if filename.endswith('.png') or filename.endswith('.jpg'):
#         # 读取图像
#         img_path = os.path.join(image_dir, filename)
#         image_name = os.path.splitext(filename)[0]
#         print(image_name)
#         # 读取相应的 world_matrix
#         matrix_path = os.path.join(image_dir, filename.replace('.png', '.npy').replace('.jpg', '.npy'))
#         world_matrix = np.load(matrix_path)
#         rotation_matrix, translation_vector = opengl_to_colmap(world_matrix)
#         # # COLMAP 的 images 表需要存储图像的名称和位姿
#         # cursor.execute('''
#         #     INSERT INTO images (name, camera_id) 
#         #     VALUES (?, ?)
#         # ''', (image_name, camera_id))

#         # image_id = cursor.lastrowid

#         # # 插入位姿信息到 image_poses 表（或其他相应的表格）
#         # cursor.execute('''
#         #     INSERT INTO image_poses (image_id, rotation, translation)
#         #     VALUES (?, ?, ?)
#         # ''', (image_id, np.array(rotation_matrix, dtype=np.float64).tobytes(), np.array(translation_vector, dtype=np.float64).tobytes()))

#         db.add_image(image_path, camera_id, rotation_matrix, translation_vector)


# # 提交更改并关闭数据库连接
# conn.commit()
# conn.close()

# # 逐帧插入图像和相机位姿信息
# for i in range(num_images):
#     image_name = f'image_{i+1:03d}.jpg'
#     image_path = os.path.join(image_dir, image_name)

#     # 获取相机的外参矩阵 world_matrix
#     # 将 OpenGL 的 world_matrix 转换为 COLMAP 所需的格式
#     rotation_matrix, translation_vector = opengl_to_colmap(world_matrix[i])

#     # 将相机位姿存入数据库
#     db.add_image(image_path, camera_id, rotation_matrix, translation_vector)

# =================================================
# # 创建COLMAP数据库
# db = pycolmap.Database(database_path)

# # 提取特征
# pycolmap.extract_features(database_path, image_dir)

# # 进行特征匹配
# pycolmap.match_exhaustive(database_path)

# # 执行稀疏重建
# reconstruction = pycolmap.Reconstruction()
# reconstruction.incremental_mapper(database_path, image_dir, output_dir)

# # 保存和可视化稀疏重建结果
# reconstruction.write(output_dir)
