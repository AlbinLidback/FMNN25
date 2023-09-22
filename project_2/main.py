import numpy as np
import matplotlib.pyplot as plt

# Functions


def rosenbrock(x):
    return 100*(x[2] - x[1]**2)**2 + (1 - x[1])**2


def test(x):
    return x[0]**2 + 4*x[1]**2 - 4*x[0]*x[1]

#


def finite_difference_gradient(x, epsilon=1e-5):
    n = len(x)
    gradient = np.zeros(n)

    for i in range(n):
        x_plus_epsilon = x.copy()
        x_plus_epsilon[i] += epsilon

        f_x = function(x)
        f_x_plus_epsilon = function(x_plus_epsilon)

        gradient[i] = (f_x_plus_epsilon - f_x) / epsilon

    return gradient


def finite_difference_hessian(x, epsilon=1e-5):
    n = len(x)
    hessian = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x_plus_epsilon_i = x.copy()
            x_plus_epsilon_j = x.copy()
            x_plus_epsilon_i[i] += epsilon
            x_plus_epsilon_j[j] += epsilon

            f_x = function(x)
            f_x_plus_epsilon_i = function(x_plus_epsilon_i)
            f_x_plus_epsilon_j = function(x_plus_epsilon_j)

            hessian[i][j] = (f_x_plus_epsilon_i - f_x_plus_epsilon_i -
                             f_x_plus_epsilon_j + f_x) / (epsilon**2)

    return hessian


def newtons_method(starting_point, tol=1e-6, max_iterations=100):
    x = np.array(starting_point)
    iteration = 0

    while iteration < max_iterations:
        gradient = finite_difference_gradient(x)
        hessian = finite_difference_hessian(x)

        # Apply symmetrizing step to Hessian matrix
        symmetrized_hessian = 0.5 * (hessian + hessian.T)

        # Compute the Newton step
        newton_step = np.linalg.solve(symmetrized_hessian, -gradient)

        # Update the current point
        x += newton_step

        # Check for convergence
        if np.linalg.norm(newton_step) < tol:
            break

        iteration += 1

    return x, function(x)


def function(x):
    return rosenbrock(x)


if __name__ == "__main__":
    initial_guess = [0.6, 1.5, 1.1]
    minimum_point, minimum_value = newtons_method(initial_guess)

    print(f"Minimum point: {minimum_point}")
    print(f"Minimum value: {minimum_value}")
