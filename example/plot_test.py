# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# # Load the 3D points from the .npy file
# points_3d = np.load('project_directory/sparse/points_3d.npy')

# # Extract x, y, z coordinates
# x = points_3d[:, 0]
# y = points_3d[:, 1]
# z = points_3d[:, 2]

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the points
# ax.scatter(x, y, z, c='r', marker='o')

# # Set labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# # Show the plot
# plt.show()

import plotly.graph_objects as go
import numpy as np

# Load the 3D points from the .npy file
points_3d = np.load('project_directory/sparse/points_3d.npy')

# Extract x, y, z coordinates
x = points_3d[:, 0]
z = points_3d[:, 1]
y = points_3d[:, 2]

# Create a scatter plot using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        size=2,
        color='red', # Color of points
    )
)])

# Set labels and plot
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    title="3D Points Visualization"
)

fig.show()
