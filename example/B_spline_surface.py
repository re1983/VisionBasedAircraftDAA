import numpy as np
from scipy.interpolate import RectBivariateSpline, bisplrep, bisplev
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
import plotly.graph_objects as go
import plotly.io as pio

# 加载 3D 点数据
points_3d = np.load('project_directory/sparse/points_3d_new.npy')

# Sample 3D points (replace with your data)
# points_3d = np.load('path_to_your_3d_points.npy')
x = points_3d[:, 0]
z = points_3d[:, 1]
y = points_3d[:, 2]

mask = (z >= -30) & (z <= 500)
x = x[mask]
y = y[mask]
z = z[mask]

# Define the grid resolution
num_bins = 50  # Adjust the resolution as needed

# Define uniform grid edges for x and y
x_min, x_max = np.min(x), np.max(x)
print(x_min, x_max)
y_min, y_max = np.min(y), np.max(y)
print(y_min, y_max)


x_edges = np.linspace(x_min, x_max, num_bins)
y_edges = np.linspace(y_min, y_max, num_bins)

# Bin Z values into a grid, using mean z-values in each bin
z_grid, x_edges, y_edges, _ = binned_statistic_2d(x, y, z, statistic='max', bins=[x_edges, y_edges])
z_grid = np.nan_to_num(z_grid, nan=-30)

# Calculate the grid centers for x and y
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
print(x_centers.shape, y_centers.shape, z_grid.shape)

# Remove NaN values (fill empty bins with the mean z-value)
z_grid = np.nan_to_num(z_grid, nan=np.nanmean(z_grid))

# Fit a B-spline to the z-values on the grid
spline = RectBivariateSpline(x_centers, y_centers, z_grid, kx=3, ky=3, s=0)
control_z = spline(x_centers, y_centers)

# Generate a dense grid for smoother plotting
Y, X = np.meshgrid(y_centers, x_centers)  # Notice Y, X order here
# print(X.shape, Y.shape)
# print(X)
Z = spline(x_centers, y_centers)

# x_1 = np.arange(x_min, x_max, 50)
# y_1 = np.arange(y_min, y_max, 50)
# z_1, _, _, _ = binned_statistic_2d(x, y, z, statistic='max', bins=[x_1, y_1])
# z_1 = np.nan_to_num(z_1, nan=0)
# # control_z = spline(x_1, y_1)
# Y_1, X_1 = np.meshgrid(y_1, x_1)  # Notice Y, X order here
# spline = RectBivariateSpline(x_1, y_1, z_1)

# num_bins = 5
# x_fine = np.linspace(np.min(x), np.max(x), 200)
# y_fine = np.linspace(np.min(y), np.max(y), 200)
# x_2 = np.arange(np.min(x), np.max(x), 50)
# y_2 = np.arange(np.min(y), np.max(y), 50)
# print(x_2.shape, y_2.shape)
# Y_2, X_2 = np.meshgrid(y_2, x_2)
# # x_edges = np.linspace(x_min, x_max, num_bins)
# # y_edges = np.linspace(y_min, y_max, num_bins)
# # x_centers = (x_edges[:-1] + x_edges[1:]) / 2
# # y_centers = (y_edges[:-1] + y_edges[1:]) / 2
# # Y_fine, X_fine = np.meshgrid(y_centers, x_centers)
# # print(X_fine.shape, Y_fine.shape)
# # print(X_fine)
# # Z_fine = spline(y_centers, x_centers)
# Z_2 = spline(Y_2, X_2)



# Create a 3D surface plot
surface_1 = go.Surface(x=X, y=Y, z=control_z, colorscale='gray', opacity=0.7)
# surface_1 = go.Surface(x=x_fine, y=y_fine, z=Z_fine, colorscale='Hot', opacity=0.7)
# surface_2 = go.Surface(x=X_2, y=Y_2, z=Z_2, colorscale='Hot', opacity=0.7)

# Create a scatter plot for thimport plotly.io as pioe original points
scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2, color='red'))

scatter_cp = go.Scatter3d(x=X.flatten(), y=Y.flatten(), z=control_z.flatten(), mode='markers', marker=dict(size=2, color='green')) # , name='Control Points
# Create the figure and add the plots
fig = go.Figure(data=[surface_1, scatter,scatter_cp])
max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min())
mid_x = (x.max() + x.min()) / 2
mid_y = (y.max() + y.min()) / 2
mid_z = (z.max() + z.min()) / 2

# Set labels
fig.update_layout(scene=dict(
    xaxis_title='X',
    yaxis_title='Y',
    zaxis_title='Z',
    xaxis=dict(range=[mid_x - max_range/2, mid_x + max_range/2], dtick=250),
    yaxis=dict(range=[mid_y - max_range/2, mid_y + max_range/2], dtick=250),
    zaxis=dict(range=[mid_z - max_range/2, mid_z + max_range/2], dtick=250)
))

# Show the plot
fig.show()
pio.write_html(fig, file="3d_surface_plot.html", auto_open=True)

