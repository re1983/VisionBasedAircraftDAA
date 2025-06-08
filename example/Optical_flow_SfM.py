import numpy as np
import cv2
import os
# from mpl_toolkits.mplot3d import Axes3D

orb = cv2.ORB_create(nfeatures = 512, scaleFactor = 2.0, nlevels = 8, 
        edgeThreshold = 31, firstLevel = 0, WTA_K = 2, scoreType = cv2.ORB_FAST_SCORE, 
        patchSize = 31, fastThreshold = 30)

def projection_matrix_to_intrinsics(projection_matrix, width, height):
    """
    从 X-Plane/OpenGL 的 projection_matrix_3d 中提取内参矩阵，结合图像分辨率 (width, height)
    """
    fx = projection_matrix[0, 0] * width / 2.0
    fy = projection_matrix[1, 1] * height / 2.0
    cx = width * (1.0 + projection_matrix[0, 2]) / 2.0
    cy = height * (1.0 - projection_matrix[1, 2]) / 2.0

    return fx, fy, cx, cy

def convert_xplane_to_opencv(projection_matrix_3d, world_matrix, width, height):
    """
    将 X-Plane 11 的 projection_matrix_3d 和 world_matrix 转换为 OpenCV 所需的投影矩阵
    """
    # 1. 从 projection_matrix_3d 提取内参矩阵 K
    fx, fy, cx, cy = projection_matrix_to_intrinsics(projection_matrix_3d, width, height)
    
    # 内参矩阵 K
    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]])
    
    # 2. 从 world_matrix 提取外参矩阵 [R | t]
    # R 是旋转矩阵 (world_matrix 的 3x3 左上部分)，t 是平移向量 (world_matrix 的前 3 个元素的第四列)
    R = world_matrix[:3, :3]  # 旋转矩阵
    t = world_matrix[:3, 3]   # 平移向量

    # 组合外参矩阵 [R | t]
    extrinsics = np.hstack((R, t.reshape(-1, 1)))

    # 3. 计算 OpenCV 投影矩阵 P = K * [R | t]
    proj_matrix_cv = np.dot(K, extrinsics)

    return proj_matrix_cv


def triangulate_all_images(projection_matrices, images):

    # 提取第一个图像的特征点
    # feature_detector = cv2.SIFT_create()  # 使用 SIFT 特征检测
    # kp1, des1 = feature_detector.detectAndCompute(images[0], None)
    kp1 = orb.detect(images[0], None)
    # 用于保存 3D 点的列表
    points_3d = []

    # 用于保存特征点的列表（关键点）
    p0 = np.array([kp.pt for kp in kp1], dtype=np.float32).reshape(-1, 1, 2)

    for i in range(1, len(images)):
        print(f"Processing image {i}...")
        # 提取下一张图像的特征
        # kp2, des2 = feature_detector.detectAndCompute(images[i], None)
        kp2 = orb.detect(images[1], None)

        # 用光流跟踪特征点
        tracked_points, st, err = cv2.calcOpticalFlowPyrLK(images[i - 1], images[i], p0, None)

        # 获取有效的点（成功跟踪的点）
        valid_pts1 = p0[st == 1]
        valid_pts2 = tracked_points[st == 1]

        # 执行三角化，P1 和 P2 是相邻帧的投影矩阵
        P1 = projection_matrices[i - 1]
        P2 = projection_matrices[i]

        # 三角化这些点
        points_4d = cv2.triangulatePoints(P1, P2, valid_pts1.T, valid_pts2.T)

        # 将齐次坐标转换为 3D 坐标
        points_3d.extend((points_4d[:3] / points_4d[3]).T)

    return np.array(points_3d)


# # 示例输入
# projection_matrix_3d = np.array([
#     [1.810660, 0, 0, 0],
#     [0, 2.414214, 0, 0],
#     [0, 0, -1.002002, -0.2002002],
#     [0, 0, -1, 0]
# ])

# world_matrix = np.array([
#     [1, 0, 0, 1],
#     [0, 1, 0, 2],
#     [0, 0, 1, 3],
#     [0, 0, 0, 1]
# ])

# width = 800  # 图像宽度
# height = 600  # 图像高度

# # 转换为 OpenCV 的投影矩阵
# opencv_proj_matrix = convert_xplane_to_opencv(projection_matrix_3d, world_matrix, width, height)

# print("OpenCV Projection Matrix: \n", opencv_proj_matrix)
# 读取投影矩阵和世界矩阵

# projection_matrix_3d = np.load('project_directory/input/projection_matrix_3d.npy')
# print(projection_matrix_3d)
# # 示例 world_matrix，可以根据需要修改
# world_matrix = np.array([
#     [1, 0, 0, 1],
#     [0, 1, 0, 2],
#     [0, 0, 1, 3],
#     [0, 0, 0, 1]
# ])

# width = 800  # 图像宽度
# height = 600  # 图像高度


# print("OpenCV Projection Matrix: \n", opencv_proj_matrix)

# 读取图像和相应的 world_matrix
image_dir = 'project_directory/test'
projection_matrix_3d = np.load('project_directory/input/projection_matrix_3d.npy')

images = []
projection_matrices = []

for filename in sorted(os.listdir(image_dir)):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # 读取图像
        img_path = os.path.join(image_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        images.append(image)
        # print(filename)
        # 读取相应的 world_matrix
        matrix_path = os.path.join(image_dir, filename.replace('.png', '.npy').replace('.jpg', '.npy'))
        world_matrix = np.load(matrix_path)
        # print(matrix_path)
        # print(world_matrix)
        # 生成 OpenCV 投影矩阵
        width, height = image.shape[1], image.shape[0]
        opencv_proj_matrix = convert_xplane_to_opencv(projection_matrix_3d, world_matrix, width, height)
        projection_matrices.append(opencv_proj_matrix)

points_3d = triangulate_all_images(projection_matrices, images)
# 显示 3D 点
# print("3D Points: \n", points_3d)
print("Number of 3D Points: ", len(points_3d))
# 保存 3D 点到文件
np.save('project_directory/sparse/points_3d_now.npy', points_3d)

# # 使用 Matplotlib 绘制 3D 点
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 提取 x, y, z 坐标
# x = points_3d[:, 0]
# y = points_3d[:, 1]
# z = points_3d[:, 2]

# ax.scatter(x, y, z, c='r', marker='o')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()

