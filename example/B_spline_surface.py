import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

# 加载 3D 点数据
points_3d = np.load('project_directory/sparse/points_3d.npy')

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
y_min, y_max = np.min(y), np.max(y)
x_edges = np.linspace(x_min, x_max, num_bins)
y_edges = np.linspace(y_min, y_max, num_bins)

# Bin Z values into a grid, using mean z-values in each bin
z_grid, x_edges, y_edges, _ = binned_statistic_2d(x, y, z, statistic='mean', bins=[x_edges, y_edges])

# Calculate the grid centers for x and y
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2

# Remove NaN values (fill empty bins with the mean z-value)
z_grid = np.nan_to_num(z_grid, nan=np.nanmean(z_grid))

# Fit a B-spline to the z-values on the grid
spline = RectBivariateSpline(x_centers, y_centers, z_grid, kx=3, ky=3, s=0)

# Generate a dense grid for smoother plotting
X, Y = np.meshgrid(x_centers, y_centers)
Z = spline(x_centers, y_centers)

import plotly.graph_objects as go

# Create a 3D surface plot
surface = go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.7)

# Create a scatter plot for the original points
scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2, color='red'))

# Create the figure and add the plots
fig = go.Figure(data=[surface, scatter])
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

