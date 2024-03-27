''' 
    Compare estimations of Data Shared Lasso and Magging on artificial data.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cvxopt import matrix, solvers
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.linalg import block_diag
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

np.random.seed(seed=0)

# Initialize alphas and ratios for Magging and Data-Shared Lasso - log-space 
alphas = np.exp(np.linspace(np.log(0.0001),np.log(5), 75))
ratios = np.exp(np.linspace(np.log(0.001), np.log(5), 10))

betas = []
beta_hats = []
f_hats = []

n_sources = 4
n_samples = 50*n_sources
group_size = int(n_samples/n_sources)
X = np.random.normal(size=(n_samples, 2))
y = np.array([])
model = Lasso(max_iter=100000)
# choose best alpha
for source in range(n_sources):
    betas.append(np.random.normal(loc=[group_size, 4], size=2))
    X_group = X[source*group_size:(source+1)*group_size]
    y_group = np.matmul(X_group,betas[source]).reshape(group_size,1) + np.random.normal(size=(int(n_samples/n_sources),1))
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

# Solve the quadratic program to obtain magging weights
f_hats = np.matrix([f_hats[source][0] for source in range(n_sources)]).T
H = f_hats.T @ f_hats / n_samples
if not all(np.linalg.eigvals(H) > 0): # Ensure H is positive definite
    print("Warning: Matrix H is not positive definite")
    H += 1e-5
P = matrix(H)
q = matrix(np.zeros(n_sources))
G = matrix(-np.eye(n_sources))
h = matrix(np.zeros(n_sources))
A = matrix(1.0, (1, n_sources))
b = matrix(1.0)
solution = solvers.qp(P, q, G, h, A, b)
w = np.array(solution['x']).round(4).flatten() # Magging weights

# Plot the convex hull of the true coefficients versus the estimated convex hull
betas_points = np.vstack([betas[source] for source in range(n_sources)])
hull_betas = ConvexHull([betas[source] for source in range(n_sources)])
fig, ax = plt.subplots()
convex_hull_plot_2d(hull_betas, ax) # true convex hull
vertices = np.matrix([betas_points[vertice] for vertice in hull_betas.vertices])
ax.fill([vertices[i,0] for i in range(len(hull_betas.vertices))], [vertices[i,1] for i in range(len(hull_betas.vertices))], alpha=0.4, label='True')
beta_hats_points = np.vstack([beta_hats[source] for source in range(n_sources)])
hull_beta_hats = ConvexHull(beta_hats_points)
convex_hull_plot_2d(hull_beta_hats, ax) # estimated convex hull
vertices = np.matrix([beta_hats_points[vertice] for vertice in hull_beta_hats.vertices])
ax.fill([vertices[i,0] for i in range(len(hull_beta_hats.vertices))], [vertices[i,1] for i in range(len(hull_beta_hats.vertices))], alpha=0.4, label='Estimate')

# Add the magging coefficient estimate
b_magging = np.dot(w,beta_hats_points)
ax.plot(b_magging[0],b_magging[1], 'or', label = 'Magging Estimator')
ax.set(xlim=(49,52.5), ylim=(2,6))
ax.legend()

# Calculate the DSL estimated coefficient for different regularization parameters (alphas) and different degrees of sharing (ratios)
_dsl_coef = []
best_dsl_mse = float('inf')
dsl_best_ratio = 0
dsl_best_alpha = 0
results = pd.DataFrame(columns=["ratio", "alpha", "mse", "coef0", "coef1", "L1 Norm"], index=range(len(ratios)*len(alphas)))
idx = 0
for ratio in ratios: 
    diag = ratio*block_diag(X[0*group_size:(0+1)*group_size], X[1*group_size:(1+1)*group_size], X[2*group_size:(2+1)*group_size], X[3*group_size:(3+1)*group_size])
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

# Add the different DSL estimates to the plot
# Estimates with the same degree of sharing should have the same colour
for idx in range(len(results)):
    r = np.log10(results.iloc[idx]["ratio"])
    color_intensity = max(0, 1 - r/10)  # Ensure color intensity is within [0, 1] range
    ax.plot(results.iloc[idx]["coef0"], results.iloc[idx]["coef1"], "o", color=(0, color_intensity * (r+3)/5, color_intensity * (2 - r)/10), alpha=0.1)

# DSL behavior when following the recommended r=1/sqrt(G)
diag = (1/np.sqrt(n_sources))*block_diag(X[0*group_size:(0+1)*group_size], X[1*group_size:(1+1)*group_size], X[2*group_size:(2+1)*group_size], X[3*group_size:(3+1)*group_size])
augmented_X = np.hstack((X, diag))
for alpha in alphas: 
    model = Lasso(alpha, max_iter=10000000)
    model.fit(augmented_X, y)
    model_mse = mean_squared_error(model.predict(augmented_X), y)
    ax.plot(model.coef_[0], model.coef_[1], "ok", alpha=0.4)

plt.savefig('images/ComparisonDSLMagging/magging_dsl_comparison.png')
plt.show()