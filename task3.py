from module_ass5 import *
import numpy as np

# set Butcher tableau for Euler method
alpha_euler = np.array([1.])
beta_euler = np.array([0.])
gamma_euler = np.array([[0.]])

# rest of your code goes here

# Using the function already coded, with (0,4) as the starting value

# myValue = explicit_rk_step(derivative_ode1, 0, 4, 0.5, [0.5, 0.5], [0, 1], np.array([[0, 0], [1, 0]]))
# print(myValue)
myValue = explicit_rk_solver(derivative_ode1, [0, 2], 4, 1, [0.5, 0.5], [0, 1], np.array([[0, 0], [1, 0]]))
print(myValue)
properValue = improved_euler_step(derivative_ode1, 0, 4, 1)
properValue = improved_euler_step(derivative_ode1, 1, properValue, 1)
print(properValue)
