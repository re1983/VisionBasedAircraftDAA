import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import bisplrep, bisplev

# 加载 3D 点数据
points_3d = np.load('project_directory/sparse/points_3d.npy')

# 提取 x, y, z 坐标
x = points_3d[:, 0]
z = points_3d[:, 1]
y = points_3d[:, 2]

sample_size = 3000  # 设置一个较小的采样数量
indices = np.random.choice(len(x), size=sample_size, replace=False)

x_sample = x[indices]
y_sample = y[indices]
z_sample = z[indices]

# 使用采样后的数据进行 B-Spline 拟合
tck = bisplrep(x_sample, y_sample, z_sample, s=1e3, kx=3, ky=3)

# # 创建规则网格，用于曲面评估
x_grid = np.linspace(np.min(x), np.max(x), 100)
y_grid = np.linspace(np.min(y), np.max(y), 100)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

# # 使用 B-Spline 拟合曲面数据
# # 3 是 B-Spline 的阶数，可以调节以改变拟合的平滑程度
# tck = bisplrep(x, y, z, s=1e3, kx=3, ky=3)

# 使用拟合参数评估曲面
z_fit = bisplev(x_grid, y_grid, tck)

# 创建 B-Spline 曲面
surface = go.Surface(x=x_mesh, y=y_mesh, z=z_fit, colorscale='Viridis')

# 创建 3D 散点图显示原始点云
scatter = go.Scatter3d(
    x=x, y=y, z=z, mode='markers', 
    marker=dict(size=2, color='red', opacity=0.8)
)

# 创建 Plotly 图表
fig = go.Figure(data=[surface, scatter])

# 设置布局和轴标签
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectratio=dict(x=1, y=1, z=0.5)
    ),
    title="3D Point Cloud B-Spline Surface Fitting"
)

# 显示图形
fig.show()
