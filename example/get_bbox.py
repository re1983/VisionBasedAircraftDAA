import numpy as np

def get_rotation_matrix(heading, pitch, roll):
	# Convert angles to radians
	heading = np.radians(heading)
	pitch = np.radians(pitch)
	roll = np.radians(roll)

	# Rotation matrices around the x, y, and z axes
	Rx = np.array([[1, 0, 0],
				   [0, np.cos(pitch), -np.sin(pitch)],
				   [0, np.sin(pitch), np.cos(pitch)]])

	Ry = np.array([[np.cos(roll), 0, np.sin(roll)],
				   [0, 1, 0],
				   [-np.sin(roll), 0, np.cos(roll)]])

	Rz = np.array([[np.cos(heading), -np.sin(heading), 0],
				   [np.sin(heading), np.cos(heading), 0],
				   [0, 0, 1]])

	# Combined rotation matrix
	R = Rz @ Ry @ Rx
	return R

def project_to_2d(point, fov, image_width, image_height):
	# Perspective projection
	f = image_width / (2 * np.tan(np.radians(fov) / 2))
	x = f * point[0] / point[2] + image_width / 2
	y = f * point[1] / point[2] + image_height / 2
	return np.array([x, y])

def get_bounding_box(length, width, height, heading, pitch, roll, distance, fov, image_width, image_height, camera_heading, camera_pitch, camera_roll):
	# Define the 8 vertices of the cube
	vertices = np.array([[length/2, width/2, height/2],
						 [length/2, width/2, -height/2],
						 [length/2, -width/2, height/2],
						 [length/2, -width/2, -height/2],
						 [-length/2, width/2, height/2],
						 [-length/2, width/2, -height/2],
						 [-length/2, -width/2, height/2],
						 [-length/2, -width/2, -height/2]])

	# Rotate the vertices according to the object's heading, pitch, and roll
	R_obj = get_rotation_matrix(heading, pitch, roll)
	rotated_vertices = vertices @ R_obj.T

	# Translate the vertices according to the object's distance
	translated_vertices = rotated_vertices + np.array([0, 0, distance])

	# Rotate the vertices according to the camera's heading, pitch, and roll
	R_cam = get_rotation_matrix(camera_heading, camera_pitch, camera_roll)
	final_vertices = translated_vertices @ R_cam.T

	# Project the vertices to 2D
	projected_vertices = np.array([project_to_2d(v, fov, image_width, image_height) for v in final_vertices])

	# Calculate the bounding box
	min_x = np.min(projected_vertices[:, 0])
	max_x = np.max(projected_vertices[:, 0])
	min_y = np.min(projected_vertices[:, 1])
	max_y = np.max(projected_vertices[:, 1])

	return min_x, min_y, max_x, max_y

# Example usage
length = 2
width = 2
height = 2
heading = 30
pitch = 15
roll = 45
distance = 10
fov = 90
image_width = 1920
image_height = 1080
camera_heading = 0
camera_pitch = 0
camera_roll = 0

bbox = get_bounding_box(length, width, height, heading, pitch, roll, distance, fov, image_width, image_height, camera_heading, camera_pitch, camera_roll)
print("Bounding box:", bbox)