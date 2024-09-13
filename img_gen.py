import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Number of points in each dimension
num_points = 90  # Adjust this value for denser point cloud

# Generate points on a grid over the surface of the half sphere
u = np.linspace(0, np.pi, num_points)
v = np.linspace(0, 2 * np.pi, num_points)
u, v = np.meshgrid(u, v)
x = np.sin(u) * np.cos(v)
y = np.sin(u) * np.sin(v)
z = -np.abs(np.cos(u))  # Negative z-coordinates for the lower half of the sphere

# Calculate gradient colors based on z-coordinates
colors = z.ravel()

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=colors, cmap='viridis', marker='o', s=15)  # Increased point size

# Set plot properties
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set limits and aspect ratio
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 0)  # Adjusted z-limits for the lower half of the sphere
ax.set_box_aspect([1,1,1])  # Equal aspect ratio for x, y, z axes

# Remove the grid
ax.grid(False)

# Remove tick labels on the axes
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# Remove tick labels on the color bar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Z-coordinate')
cbar.ax.set_yticklabels([])

# Define vertices of the tilted plane
plane_x = [-1, 1, 1, -1]
plane_y = [0, 0, 1, 1]  # Adjusted y-coordinates to create a 60-degree angle with the z-axis
plane_z = [0, 0, -0.4, -0.4]
verts = [list(zip(plane_x, plane_y, plane_z))]

# Add transparent black plane to the plot
plane = Poly3DCollection(verts, alpha=0.8, facecolor='k', edgecolor='none')
ax.add_collection3d(plane)

# Save the plot as a PNG file
plt.savefig('gradient_lower_half_sphere_with_tilted_plane.png')

# Show the plot (optional)
plt.show()
