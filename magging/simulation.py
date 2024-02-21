''' This script calculates the magging estimator for an artifical dataset and plots it (2D). Problem: X is wrong - we need np.identity(2) in order to be useful
'''

import pandas as pd

from magging import Magging
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy.linalg import block_diag

'''X = np.identity(3) # predictor matrix 

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
'''

######## Random comparison DSL, Magging

np.random.seed(seed=0)

alphas = np.exp(np.linspace(np.log(0.0001),np.log(5), 75))
ratios = np.exp(np.linspace(np.log(0.001), np.log(5), 5))

betas = []
beta_hats = []
f_hats = []

n_sources = 4
n_samples = 5*n_sources
X = np.random.normal(size=(n_samples, 2))
y = np.array([])
model = Lasso(max_iter=100000)
######## choose best alpha
for source in range(n_sources):
    betas.append(np.random.normal(loc=[5, 4], size=2))
    X_group = X[source*5:(source+1)*5]
    y_group = np.matmul(X_group,betas[source]).reshape(5,1) + np.random.normal(size=(int(n_samples/n_sources),1))
    y = np.append(y, y_group)
    best_alpha = 0
    best_mse = float('inf')
    for alpha in alphas:
        model.alpha = alpha
        model.fit(X_group, y_group)
        model_mse = mean_squared_error(model.predict(X_group), y_group)
        if best_mse > model_mse: 
            best_alpha = alpha
            best_mse = model_mse
    model.alpha = best_alpha
    model.fit(X_group,y_group)
    print('group: ', source ,' best alpha: ', best_alpha, ' best_mse: ', best_mse)
    beta_hats.append(model.coef_)
    y_pred = model.predict(X).reshape((1,n_samples))
    print(y_pred)
    f_hats.append(y_pred)

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

fig, ax = plt.subplots()
convex_hull_plot_2d(hull_betas, ax)
vertices = np.matrix([betas_points[vertice] for vertice in hull_betas.vertices])
ax.fill([vertices[i,0] for i in range(len(hull_betas.vertices))], [vertices[i,1] for i in range(len(hull_betas.vertices))], alpha=0.4, label='True')
beta_hats_points = np.vstack([beta_hats[source] for source in range(n_sources)])
hull_beta_hats = ConvexHull(beta_hats_points)
convex_hull_plot_2d(hull_beta_hats, ax)
vertices = np.matrix([beta_hats_points[vertice] for vertice in hull_beta_hats.vertices])
ax.fill([vertices[i,0] for i in range(len(hull_beta_hats.vertices))], [vertices[i,1] for i in range(len(hull_beta_hats.vertices))], alpha=0.4, label='Estimate')
b_magging = np.dot(w,beta_hats_points)
print(b_magging)
ax.plot(b_magging[0],b_magging[1], 'or', label = 'Magging Estimator')
ax.set(xlim=(2,8), ylim=(2,6))
ax.legend()

_dsl_coef = []
best_dsl_mse = float('inf')
dsl_best_ratio = 0
dsl_best_alpha = 0

results = pd.DataFrame(columns=["ratio", "alpha", "mse", "coef0", "coef1", "L1 Norm"], index=range(len(ratios)*len(alphas)))
idx = 0

for ratio in ratios: 
    diag = ratio*block_diag(X[0*5:(0+1)*5], X[1*5:(1+1)*5], X[2*5:(2+1)*5], X[3*5:(3+1)*5])
    augmented_X = np.hstack((X, diag))
    for alpha in alphas: 
        model = Lasso(alpha, max_iter=10000000)
        model.fit(augmented_X, y)
        model_mse = mean_squared_error(model.predict(augmented_X), y)
        if best_dsl_mse > model_mse:
            best_dsl_mse = model_mse
            dsl_best_ratio = ratio
            dsl_best_alpha = alpha
        _dsl_coef.append(model.coef_[:2])

        results.iloc[idx] = (ratio, alpha, model_mse, model.coef_[0], model.coef_[1], np.abs(model.coef_[0])+np.abs(model.coef_[1]))
        idx +=1

df = pd.DataFrame(results)

# mask = np.abs(results["ratio"] - 10) <= 1e-6
# results = results[mask]
print(df[df['L1 Norm'] <= 1e-6])

# for coef in _dsl_coef:
#     ax.plot(coef[0], coef[1], 'og', alpha = 0.05)

for idx in range(len(results)):
    r = np.log10(results.iloc[idx]["ratio"])
    ax.plot(results.iloc[idx]["coef0"], results.iloc[idx]["coef1"], "o", color=(0, (r+3)/5, (2 - r)/10), alpha=0.3)

plt.show()