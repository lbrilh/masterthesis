''' This script calculates the magging estimator for an artifical dataset and plots it (2D).
'''



from magging import Magging
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


X = np.identity(3) # predictor matrix 

B = np.matrix('0.5 1.5 2; 1 0.5 1') # matrix containing group coefficient estimators

fhat = np.matrix('0.5 1.5 2; 1 0.5 1') # fhat == B 

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
print(w)
print()
# Your matrix B
B = np.matrix('0.5 1.5 2; 1 0.5 1')
# Create a figure and axis for the plot
fig, ax = plt.subplots()

ax.plot([B[0,0],B[0,1]], [B[1,0],B[1,1]], '-ok')
ax.annotate('b_1', xy=(.35,1))
ax.plot([B[0,0],B[0,2]], [B[1,0],B[1,2]], '-ok')
ax.annotate('b_2', xy=(1.55,0.5))
ax.plot([B[0,1],B[0,2]], [B[1,1],B[1,2]], '-ok')
ax.annotate('b_3', xy=(2,1.05))
ax.plot([0,0.5], [0,1], 'r-')
ax.plot(0.5,1, 'or')
ax.annotate('b_maximin', xy=(0.5,1.05), color='red')
ax.set(xlim=(-0,2.5), ylim=(0,1.5))
ax.spines[['top','right']].set_visible(False)
ax.plot(0.5,0.5, 'ob')
ax.annotate('b_DSL', xy=(0.5,0.55), color='blue')
plt.fill([0.5,1.5,2],[1,0.5,1], color='k', alpha=0.2)
plt.xticks([0,1,2])
plt.yticks([0,0.5,1])
plt.show()