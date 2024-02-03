import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# Define your points
points = np.array([[0.0, 13.55973753],
                   [-2.4220879, 13.83846328],
                   [0.26937025, 13.0436701],
                   [0.54201526, 13.61477594]])

# Calculate the convex hull
hull = ConvexHull(points)

# The vertices of the convex hull
hull_vertices = points[hull.vertices]

# Plot the points
plt.plot(points[:,0], points[:,1], 'o', label='Original Points')

# Plot the convex hull
plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2, label='Convex Hull')

# Set labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Show the plot
plt.show()
