from module_ass5 import *
from scipy.integrate import solve_ivp

# y_exact = exact_solution_ode1(np.array([0,2]))


output = solve_ivp(derivative_ode1, [0, 2], [4])
t_values = output.t
y_approx = output.y

y_exact = exact_solution_ode1(t_values)
mae = mean_absolute_error(y_exact[1:], y_approx[0][1:])
print(mae)

plt.figure(figsize=(10, 5))

plt.plot(t_values, y_approx[0], label="Approximate Solution", color='red', marker='s')
plt.plot(t_values, y_exact, label="Exact Solution", color='blue', marker='o')
plt.legend()
plt.show()
pass
