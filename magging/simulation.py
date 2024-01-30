from magging import Magging
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


X = np.identity(3)

B = np.matrix('1 1 2; 1 -1 1')

fhat = np.matrix('1 1 2; 1 -1 1') # fhat == B 

# Set-up of the quadratic program to determine magging weights
r = 3
if r == 1:
    print('Warning: Only one group exists!')
H = fhat.T @ fhat / X.shape[0]

print(np.linalg.eigvals(H))

if not all(np.linalg.eigvals(H) > 0): # Ensure H is positive definite
    print("Warning: Matrix H is not positive definite")
    H += 1e-5
P = matrix(H)
q = matrix(np.zeros(r))
G = matrix(-np.eye(r))
h = matrix(np.zeros(r))
A = matrix(1.0, (1, r))
b = matrix(1.0)

# Solve the quadratic program to obtain magging weights
solution = solvers.qp(P, q, G, h, A, b)
w = np.array(solution['x']).round(4).flatten() # Magging weights

# Your matrix B
B = np.matrix('1 1 2; 1 -1 1')

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Plot the column vectors of B as points
ax.plot(B[0], B[1], 'o', markersize=8)

# Connect the points of B with lines to form the convex hull
ax.plot(*B[:, [0, 1, 2, 0]], 'k-')  # Connecting points in sequence and back to the start

# Plot the red line (b_maximin line) along the x-axis
ax.plot([0, 1], [0, 0], 'r-')

# Find the point with the largest x-coordinate, which is on or below the red line (y <= 0)
# This point is the intersection with the convex hull
b_maximin_point = B.T[np.argmin(B[1, :])]
b_maximin_x, b_maximin_y = b_maximin_point[0, 0], b_maximin_point[0, 1]

# Plot the point b_maximin
ax.plot(b_maximin_x, b_maximin_y, 'ro')  # Red point

# Annotate the point b_maximin
ax.annotate('b_maximin', (b_maximin_x, b_maximin_y), textcoords="offset points", xytext=(5, -5))

# Set labels for the axes
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

# Show the plot
plt.show()

print(w)
raise ValueError
if isinstance(self.model, (Lasso, LinearRegression)):
    coef = []
    self.groups_magging_dist = {}
    for group in self.groups:
        model_coef = self.models[group].coef_
        coef.append(model_coef)
        self.groups_magging_dist[group] = self.magging_distance(model_coef)

    magging_coef = np.dot(np.matrix(coef).T, w).T
    self.magging_dist = self.magging_distance(magging_coef)

    print(self.groups_magging_dist)
    print(self.magging_dist)

print('Magging weights: ', w)

print(fhat)