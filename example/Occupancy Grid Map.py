import numpy as np
import plotly.graph_objects as go

# 加载 3D 点数据
points_3d = np.load('project_directory/sparse/points_3d.npy')

# 设置解析度（1x1x1 的栅格）
resolution = 1

# 获取 3D 点的最小和最大值，以确定栅格的范围
x_min, y_min, z_min = np.min(points_3d, axis=0)
x_max, y_max, z_max = np.max(points_3d, axis=0)

# 计算栅格中的点坐标（对 3D 空间进行离散化）
x_idx = ((points_3d[:, 0] - x_min) / resolution).astype(int)
y_idx = ((points_3d[:, 1] - y_min) / resolution).astype(int)
z_idx = ((points_3d[:, 2] - z_min) / resolution).astype(int)

# 使用 np.unique 来得到占据栅格的唯一坐标
occupied_voxels = np.unique(np.stack([x_idx, y_idx, z_idx], axis=1), axis=0)

# 可视化占据栅格（用立方体表示每个占据的单元）
fig = go.Figure()

# 依次为每个占据的栅格绘制立方体
for voxel in occupied_voxels:
    # 获取栅格的中心点坐标
    x, y, z = voxel * resolution + resolution / 2
    # 绘制立方体（用小立方体表示占据的栅格）
    fig.add_trace(go.Mesh3d(
        x=[x-0.5, x+0.5, x+0.5, x-0.5, x-0.5, x+0.5, x+0.5, x-0.5],
        y=[y-0.5, y-0.5, y+0.5, y+0.5, y-0.5, y-0.5, y+0.5, y+0.5],
        z=[z-0.5, z-0.5, z-0.5, z-0.5, z+0.5, z+0.5, z+0.5, z+0.5],
        color='red', opacity=0.5
    ))

# 设置布局和标签
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectratio=dict(x=1, y=1, z=1),
    ),
    title="3D Occupancy Grid Mapping"
)

# 显示图形
fig.show()
