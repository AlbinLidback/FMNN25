# import numpy as np
# from optimization_problem import *


# class Optimization_method:
#     def __init__(self, problem, method):
#         self.problem = problem
#         self.method = method


# def finite_difference_hessian(x, epsilon=1e-5):
#     n = len(x)
#     hessian = np.zeros((n, n))

#     for i in range(n):
#         for j in range(n):
#             # Compute the second partial derivative using central finite differences
#             x_plus_epsilon = x.copy()
#             x_plus_epsilon[i] += epsilon
#             x_plus_epsilon[j] += epsilon

#             x_plus_i = x.copy()
#             x_plus_i[i] += epsilon

#             x_plus_j = x.copy()
#             x_plus_j[j] += epsilon

#             f_x = objective_function(x)
#             f_x_plus_epsilon = objective_function(x_plus_epsilon)
#             f_x_plus_i = objective_function(x_plus_i)
#             f_x_plus_j = objective_function(x_plus_j)

#             hessian[i][j] = (f_x_plus_epsilon - f_x_plus_i -
#                              f_x_plus_j + f_x) / (epsilon**2)

#     return hessian


# def newton_method(objective_function, starting_point, tol=1e-6, max_iterations=100):
#     x = np.array(starting_point)
#     iteration = 0

#     while iteration < max_iterations:
#         gradient = np.gradient(objective_function(x), x)
#         hessian = finite_difference_hessian(x)

#         # Apply symmetrizing step to Hessian matrix
#         symmetrized_hessian = 0.5 * (hessian + hessian.T)

#         # Compute the Newton step
#         newton_step = np.linalg.solve(symmetrized_hessian, -gradient)

#         # Update the current point
#         x += newton_step

#         # Check for convergence
#         if np.linalg.norm(newton_step) < tol:
#             break

#         iteration += 1

#     return x, objective_function(x)
