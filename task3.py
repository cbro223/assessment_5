from module_ass5 import *
import numpy as np
import matplotlib.pyplot as plt

# set Butcher tableau for Euler method
alpha_euler = np.array([1.])
beta_euler = np.array([0.])
gamma_euler = np.array([[0.]])

# set the Butcher tableau for the improved euler method
alpha_improved_euler = np.array([0.5, 0.5])
beta_improved_euler = np.array([0, 1])
gamma_improved_euler = np.array([[0, 0], [1, 0]])

# Set the butcher tableau for the RK4 method
alpha_rk4 = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
beta_rk4 = np.array([0, 0.5, 0.5, 1])
gamma_rk4 = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])

#All the h values array
h = [0.01, 0.025, 0.05, 0.1, 0.2, 0.25]

# Initialising the MAE arrays
euler_mae = np.zeros(len(h))
improved_euler_mae = np.zeros(len(h))
rk4_mae = np.zeros(len(h))

for i in range(len(h)):
    # Estimates the using the respective methods and computes the MAE
    t, estimated_values_euler = explicit_rk_solver(derivative_ode1, [0, 2], 4, h[i], alpha_euler, beta_euler,
                                                   gamma_euler)

    proper_values = exact_solution_ode1(t)
    mae_euler = mean_absolute_error(proper_values[1:], estimated_values_euler[1:])
    euler_mae[i] = mae_euler

    t, estimated_values_improved_euler = explicit_rk_solver(derivative_ode1, [0, 2], 4, h[i], alpha_improved_euler,
                                                            beta_improved_euler, gamma_improved_euler)
    improved_euler_mae[i] = mean_absolute_error(proper_values[1:], estimated_values_improved_euler[1:])

    t, estimated_values_rk4 = explicit_rk_solver(derivative_ode1, [0, 2], 4, h[i], alpha_rk4, beta_rk4, gamma_rk4)
    rk4_mae[i] = mean_absolute_error(proper_values[1:], estimated_values_rk4[1:])

# Plots all the MAEs on a graph
plt.plot(h, euler_mae, label="Euler MAE", color="red")
plt.plot(h, improved_euler_mae, label="Improved Euler MAE", color="blue")
plt.plot(h, rk4_mae, label="RK4 MAE", color="black")
plt.title("Mean Absolute Error vs step size(h)")
plt.xlabel("Step Size")
plt.ylabel("MAE")
plt.legend()
plt.show()

# Plots the real and approximated values at h = 0.5
h = 0.5
t, euler_values = explicit_rk_solver(derivative_ode1, [0, 2], 4, h, alpha_euler, beta_euler, gamma_euler)
t, euler_i_values = explicit_rk_solver(derivative_ode1, [0, 2], 4, h, alpha_improved_euler, beta_improved_euler,
                                       gamma_improved_euler)
t, rk4_values = explicit_rk_solver(derivative_ode1, [0, 2], 4, h, alpha_rk4, beta_rk4, gamma_rk4)

plt.figure(2)
exact_values = exact_solution_ode1(t)
plt.plot(t, exact_values, label="Exact Values", color="red")
plt.plot(t, euler_values, label="Euler Values", color="blue")
plt.plot(t, euler_i_values, label="Improved Euler Values", color="black")
plt.plot(t, rk4_values, label="RK4 Values", color="Purple")

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Y against Time")
plt.legend()
plt.show()
