import numpy as np
from module_ass5 import *

# rest of your code goes here

# Using the function already coded, with (0,4) as the starting value

# myValue = explicit_rk_step(derivative_ode1, 0, 4, 0.5, [0.5, 0.5], [0, 1], np.array([[0, 0], [1, 0]]))
# print(myValue)
# myValue = explicit_rk_solver(derivative_ode1, [0, 2], 4, 1, [0.5, 0.5], [0, 1], np.array([[0, 0], [1, 0]]))
# print(myValue)
# properValue = improved_euler_step(derivative_ode1, 0, 4, 1)
# properValue = improve
# d_euler_step(derivative_ode1, 1, properValue, 1)
# print(properValue)

# Task completion



alpha_euler = np.array([1.])
beta_euler = np.array([0.])
gamma_euler = np.array([[0.]])

alpha_improved_euler = np.array([0.5, 0.5])
beta_improved_euler = np.array([0, 1])
gamma_improved_euler = np.array([[0, 0], [1, 0]])

alpha_rk4 = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
beta_rk4 = np.array([0, 0.5, 0.5, 1])
gamma_rk4 = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])

print(explicit_rk_step(derivative_ode1, 0, 4, 1, alpha_improved_euler, beta_improved_euler, gamma_improved_euler))
t, values = explicit_rk_solver(derivative_ode1, [0,1], 4, 0.01, alpha_euler, beta_euler, gamma_euler)

print(values)