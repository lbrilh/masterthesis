''' This script calculates the magging estimator for an artifical dataset and plots it (2D). Problem: X is wrong - we need np.identity(2) in order to be useful
'''



from magging import Magging
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.linear_model import LinearRegression


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
'''
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
plt.show()'''


######## Random comparison DSL, Magging

betas = []
beta_hats = []
f_hats = []

n_sources = 4
n_samples = 5*n_sources
X = np.random.normal(size=(n_samples, 2))
model = LinearRegression()
for source in range(n_sources):
    betas.append(np.random.normal(loc=[5, 4], size=2))
    y = np.matmul(X[source*5:(source+1)*5],betas[source]).reshape(5,1) + np.random.normal(size=(int(n_samples/n_sources),1))
    beta_hats.append(model.fit(X[source*5:(source+1)*5],y).coef_)
    print(model.predict(X).reshape((1,n_samples)))
    f_hats.append(model.predict(X).reshape((1,n_samples)))

print('betas: ', betas)
print()
print('beta hats: ', beta_hats)
print()
print('f_hats: ', f_hats)
print()
print(f_hats[0][0])
print('f_hats as matrix: ', np.matrix([f_hats[0][0], f_hats[1][0], f_hats[2][0]]).T)
print()

f_hats = np.matrix([f_hats[source][0] for source in range(n_sources)]).T
H = f_hats.T @ f_hats / n_samples

print(np.linalg.eigvals(H))

if not all(np.linalg.eigvals(H) > 0): # Ensure H is positive definite
    print("Warning: Matrix H is not positive definite")
    H += 1e-5
P = matrix(H)
q = matrix(np.zeros(n_sources))
G = matrix(-np.eye(n_sources))
h = matrix(np.zeros(n_sources))
A = matrix(1.0, (1, n_sources))
b = matrix(1.0)

# Solve the quadratic program to obtain magging weights
solution = solvers.qp(P, q, G, h, A, b)
w = np.array(solution['x']).round(4).flatten() # Magging weights
print(w)
betas_points = np.vstack([betas[source] for source in range(n_sources)])
hull_betas = ConvexHull([betas[source] for source in range(n_sources)])
'''fig, ax = plt.subplots()
for source in range(n_sources):
    ax.plot(betas[source][0], betas[source][1], 'ok')

ax.plot(betas_points[hull_betas.vertices,0], betas_points[hull_betas.vertices,1], '-ok', lw=2)
ax.plot([betas_points[hull_betas.vertices[0]][0], betas_points[hull_betas.vertices[-1]][0]], 
        [betas_points[hull_betas.vertices[0]][1], betas_points[hull_betas.vertices[-1]][1]], '-ok', lw=2)
plt.show()'''

fig, ax = plt.subplots()
convex_hull_plot_2d(hull_betas, ax)
beta_hats_points = np.vstack([beta_hats[source] for source in range(n_sources)])
b_magging = np.dot(w,beta_hats_points)
print(b_magging)
ax.plot(b_magging[0],b_magging[1], 'or')
ax.annotate('b_magging', xy=(b_magging[0]-0.1, b_magging[1]-0.3), color='red')
hull_beta_hats = ConvexHull(beta_hats_points)
convex_hull_plot_2d(hull_beta_hats, ax)
ax.set(xlim=(3,7), ylim=(2,6))
plt.show()