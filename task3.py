import numpy

from module_ass5 import *
import numpy as np
import matplotlib.pyplot as plt

# set Butcher tableau for Euler method
alpha_euler = np.array([1.])
beta_euler = np.array([0.])
gamma_euler = np.array([[0.]])

alpha_improved_euler = np.array([0.5, 0.5])
beta_improved_euler = np.array([0, 1])
gamma_improved_euler = np.array([[0, 0], [1, 0]])

alpha_rk4 = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
beta_rk4 = np.array([0, 0.5, 0.5, 1])
gamma_rk4 = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])

computedValue = exact_solution_ode1([0, 1, 2])
h = [0.01, 0.025, 0.05, 0.1, 0.2, 0.25]

euler_mae = np.zeros(6)
improved_euler_mae = np.zeros(6)
rk4_mae = np.zeros(6)

for i in range(len(h)):
    tspan = np.arange(0,2,h[i])

    proper_values = exact_solution_ode1(tspan)
    estimated_values_euler = explicit_rk_solver(derivative_ode1, [0,2], 4,h[i], alpha_euler, beta_euler, gamma_euler)

    mae_euler = mean_absolute_error(proper_values, estimated_values_euler)
    euler_mae[i] = mae_euler

    estimated_values_improved_euler = explicit_rk_solver(derivative_ode1, [0,2],4,h[i], alpha_improved_euler, beta_improved_euler, gamma_improved_euler)
    improved_euler_mae[i] = mean_absolute_error(proper_values, estimated_values_improved_euler)

    estimated_values_rk4 = explicit_rk_solver(derivative_ode1, [0,2], 4, h[i], alpha_rk4, beta_rk4, gamma_rk4)
    rk4_mae[i] = mean_absolute_error(proper_values, estimated_values_rk4)

plt.plot(h, euler_mae, label = "Euler MAE", color = "red")
plt.plot(h, improved_euler_mae, label = "Improved Euler MAE", color = "blue")
plt.plot(h, rk4_mae, label = "RK4 MAE", color = "black")
plt.xlabel("Step Size")
plt.ylabel("MAE")
plt.legend()
plt.show()
