import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# 加载 3D 点数据
points_3d = np.load('project_directory/sparse/points_3d.npy')

# 提取 x, y, z 坐标
x = points_3d[:, 0]
z = points_3d[:, 1]
y = points_3d[:, 2]
# # 过滤 z 小于 -30 的所有值
mask = (z >= -30) & (z <= 400)
x = x[mask]
y = y[mask]
z = z[mask]
# 创建网格，用来在该网格上插值
grid_x, grid_y = np.mgrid[np.min(x):np.max(x):100j, np.min(y):np.max(y):100j]

# 使用 SciPy 的 griddata 来进行 3D 曲面插值
grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

# 创建一个 Plotly 的 3D Surface Plot
surface = go.Surface(x=grid_x, y=grid_y, z=grid_z, colorscale='Viridis')

# 创建 3D 散点图，显示原始点云
scatter = go.Scatter3d(
    x=x, y=y, z=z, mode='markers', 
    marker=dict(size=2, color='red', opacity=0.8)
)

max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min())
mid_x = (x.max() + x.min()) / 2
mid_y = (y.max() + y.min()) / 2
mid_z = (z.max() + z.min()) / 2

# 创建 Plotly 图表
fig = go.Figure(data=[surface, scatter])
# fig = go.Figure(data=[scatter])
# 设置布局和轴标签
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        xaxis=dict(range=[mid_x - max_range/2, mid_x + max_range/2], dtick=250),
        yaxis=dict(range=[mid_y - max_range/2, mid_y + max_range/2], dtick=250),
        zaxis=dict(range=[mid_z - max_range/2, mid_z + max_range/2], dtick=250)
    ),
    title="3D Point Cloud Surface Fitting"
)

# 显示图形
fig.show()
# 保存图形为 HTML 文件
# fig.write_html('3d_surface_plot.html')
