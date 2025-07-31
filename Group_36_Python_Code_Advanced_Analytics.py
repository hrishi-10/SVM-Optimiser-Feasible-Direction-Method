# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:54:29 2025

@author: Group36
"""

# Preparation / Comment
# 1. Ensure that the Excel-CSV file "Data.csv" is saved and that the Python
#    directory is set to the correct location.
# 2. If necessary, install Ipopt:
#    -> Open the Anaconda Prompt and install the package using the following command:
#       conda install -c conda-forge ipopt

# Import and load required packages
import pyomo.environ as pyo
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

# Load dataset with custom column-titles
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv("Data.csv", names=column_names, header=None)

# Preprocess data (Extract X and Y-Values)
X = df.iloc[:, :-1].values
y = np.where(df.iloc[:, -1] == "Iris-setosa", 1, -1)
# Label Y-values with 1 if "Iris-Setosa", -1 otherwise

# Extract the shape of X
n_samples, n_features = X.shape

# Set the bound-value M
M = 10
# Initially we set M to a high value e.g. 100
# After that we interpreted the optimal solution found and decreased the value to 10

# STEP 1: PREPARE THE INITIALISATION

# Find the Closest Points from Opposite Classes
X_pos = X[y == 1]   # Samples with label 1
X_neg = X[y == -1]  # Samples with label -1

# Initialise minimum distance as infinity (Prepare placeholders: min_dist and closest_pair)
min_dist = float('inf')
closest_pair = (None, None)

# Search for the closest pair (one from each class)
for x_pos in X_pos:
    for x_neg in X_neg:
        dist = euclidean(x_pos, x_neg)  # Compute Euclidean distance
        if dist < min_dist:
            min_dist = dist
            closest_pair = (x_pos, x_neg)

# Extract the closest points
x_pos_closest, x_neg_closest = closest_pair

# Compute the Midpoint
x_mid = (x_pos_closest + x_neg_closest) / 2  # Midpoint
w_0 = x_pos_closest - x_neg_closest  # Direction vector (serves as an initial normal vector for the hyperplane)
b_0 = -np.dot(w_0, x_mid)  # w^T*x + b = 0 -> reformulated: b_0 = -w_0^T*x_mid
# Ensures the midpoint lies on the decision boundary -> providing an initial hyperplane (not necessarily optimal)
# -> We'll optimise w_0 and b_0 later with the feasible direction method

print(f"Closest points found: {x_pos_closest} and {x_neg_closest}")
print(f"Midpoint: {x_mid}")
print(f"Initial w_0: {w_0}")
print(f"Initial b_0: {b_0}")

# Check that the initialisation values are in the feasible set
feasible_w = np.all((w_0 >= -M) & (w_0 <= M))
feasible_b = -M <= b_0 <= M
feasible_constraints = np.all(y * (np.dot(X, w_0) + b_0) >= 1)

if feasible_w and feasible_b and feasible_constraints:
    print("Initial values (w_0, b_0) are in the feasible set.")
else:
    print("Initial values (w_0, b_0) are NOT in the feasible set.")

# Set up w_k & b_k for the first iteration (0) with the initialisation values
w_k = w_0.copy()
b_k = b_0

# Set iteration limits -> epsilon or max number of iterations
epsilon = 0.01
max_iter = 500

# Set up the loop that either iterates until max_iter is reached or stops as soon as 
# the stopping condition ||z_k - z_(k-1)|| ≤ ε (convergence criteria) is satisfied.
for k in range(max_iter):
    w_k_prev = w_k.copy() # Save the current w_k before updating in this iteration
    b_k_prev = b_k # Save the current b_k before updating in this iteration

    # STEP 2: Find v_k
    # Create Pyomo model
    model = pyo.ConcreteModel()
    
    # Define decision variables with bounds
    model.w_v = pyo.Var(range(n_features), domain=pyo.Reals, bounds=(-M, M))
    model.b_v = pyo.Var(domain=pyo.Reals, bounds=(-M, M))  # Add b as a variable

    # Define the objective function: min w_k^T*v_w
    model.obj = pyo.Objective(expr=sum(w_k[j] * model.w_v[j] for j in range(n_features)), sense=pyo.minimize)

    # Define SVM margin constraints: y_i (w^T x_i + b) ≥ 1
    def svm_constraint_rule(model, i):
        return y[i] * (sum(model.w_v[j] * X[i][j] for j in range(n_features)) + model.b_v) >= 1

    # Add the constraints
    model.svm_constraints = pyo.Constraint(range(n_samples), rule=svm_constraint_rule)

    # Solve the optimisation problem
    solver = pyo.SolverFactory('glpk')
    solver.solve(model)

    # Extract optimal v_k
    v_w_k_opt = [pyo.value(model.w_v[j]) for j in range(n_features)]
    v_b_k_opt = pyo.value(model.b_v)  # Extract optimised bias

    print("Optimal v_k:", v_w_k_opt)
    print("Optimal b_k:", v_b_k_opt)

    # Step 3: Find the descent direction dk
    d_w_k = np.array(v_w_k_opt) - np.array(w_k)  # Compute for weights
    d_b_k = v_b_k_opt - b_k  # Now we update b_k too
    d_k = np.append(d_w_k, d_b_k)

    print("Descent direction d_k:", d_k)

    # Step 4: Find the optimal step size tau_k
    model_tau = pyo.ConcreteModel()
    model_tau.tau = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0,1))
    model_tau.obj = pyo.Objective(
        expr=0.5 * sum((w_k[j] + model_tau.tau * d_k[j]) ** 2 for j in range(n_features)), # j goes from 0, 1, 2, 3 (with that only w will be included) 
        sense=pyo.minimize)
    
    # Solve the optimisation problem
    solver = pyo.SolverFactory('ipopt')
    solver.solve(model_tau)
    
    # Extract optimal tau_k
    tau_k_opt = pyo.value(model_tau.tau)
    print("Optimal tau_k:", tau_k_opt)

    # Step: Update z_k (w_k, b_k)
    w_k = np.array(w_k) + tau_k_opt * np.array(d_w_k)
    b_k = b_k + tau_k_opt * d_b_k

    print("Updated w_k:", w_k)
    print("Updated b_k:", b_k)

    # Stopping condition: ||z_k - z_(k-1)|| ≤ ε
    if np.linalg.norm(np.append(w_k - w_k_prev, b_k - b_k_prev)) <= epsilon:
        print("Convergence reached at iteration", k)
        break
    
    print("Iteration:", k)

# Some control outputs 

# Compute margin for each data point
margins = y * (np.dot(X, w_k) + b_k)

# Check how many satisfy y_i (w^T x_i + b) >= 1
num_satisfied = np.sum(margins >= 1)
num_violated = np.sum(margins < 1)

print(f"Margin Constraint Check:")
print(f"Number of samples satisfying constraint: {num_satisfied}/{n_samples}")
print(f"Number of violations: {num_violated}/{n_samples}")
# If num_violated > 0, it means some constraints are not satisfied.

# Compute predictions
predictions = np.sign(np.dot(X, w_k) + b_k)

# Compute accuracy
accuracy = np.mean(predictions == y)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")

# Output the final result
print(f"Final w_k: {w_k}")
print(f"Final b_k: {b_k}")