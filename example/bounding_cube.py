import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def generate_bounding_box(nose, tail, right, left, top, bottom):
    # 计算中心点
    center_x = (nose[0] + tail[0]) / 2
    center_y = (right[1] + left[1]) / 2
    center_z = (top[2] + bottom[2]) / 2

    # 计算半边长
    half_length = np.linalg.norm(np.array(nose) - np.array(tail)) / 2
    half_width = np.linalg.norm(np.array(right) - np.array(left)) / 2
    half_height = np.linalg.norm(np.array(top) - np.array(bottom)) / 2

    # 生成八个顶点
    vertices = np.array([
        [center_x + half_length, center_y + half_width, center_z + half_height],  # 前右上
        [center_x + half_length, center_y + half_width, center_z - half_height],  # 前右下
        [center_x + half_length, center_y - half_width, center_z + half_height],  # 前左上
        [center_x + half_length, center_y - half_width, center_z - half_height],  # 前左下
        [center_x - half_length, center_y + half_width, center_z + half_height],  # 后右上
        [center_x - half_length, center_y + half_width, center_z - half_height],  # 后右下
        [center_x - half_length, center_y - half_width, center_z + half_height],  # 后左上
        [center_x - half_length, center_y - half_width, center_z - half_height]   # 后左下
    ])

    return vertices

# 示例使用
nose = [10, 0, 0]
tail = [-10, 0, 0]
right = [0, 5, 0]
left = [0, -5, 0]
top = [0, 0, 3]
bottom = [0, 0, -3]

vertices = generate_bounding_box(nose, tail, right, left, top, bottom)
print("Bounding Box Vertices:")
print(vertices)
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extracting the coordinates for plotting
x = vertices[:, 0]
y = vertices[:, 1]
z = vertices[:, 2]

# Plotting the vertices
ax.scatter(x, y, z, c='r', marker='o')

# Annotating the vertices
annotations = ['Front-Right-Top', 'Front-Right-Bottom', 'Front-Left-Top', 'Front-Left-Bottom',
                'Back-Right-Top', 'Back-Right-Bottom', 'Back-Left-Top', 'Back-Left-Bottom']

for i, txt in enumerate(annotations):
    ax.text(x[i], y[i], z[i], txt)

# Setting labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
# Plotting the additional points
# additional_points = {
#     'Nose': nose,
#     'Tail': tail,
#     'Right': right,
#     'Left': left,
#     'Top': top,
#     'Bottom': bottom
# }

# for label, point in additional_points.items():
#     ax.scatter(point[0], point[1], point[2], c='b', marker='^')
#     ax.text(point[0], point[1], point[2], label)

# plt.show()