from module_ass5 import *
from scipy.integrate import solve_ivp

# Approximate the ODE using the IVP function
output = solve_ivp(derivative_ode1, [0, 2], [4])
t_values = output.t
y_approx = output.y

# Computes the MAE for the exact vs IVP methods
y_exact = exact_solution_ode1(t_values)
mae = mean_absolute_error(y_exact[1:], y_approx[0][1:])
print(mae)

plt.figure(figsize=(10, 5))
# Plot the exact and approximate values on the same plot

plt.plot(t_values, y_approx[0], label="Approximate Solution", color='red', marker='s')
plt.plot(t_values, y_exact, label="Exact Solution", color='blue', marker='o', linestyle='dashed')
plt.legend()
plt.title("Y values against time")
plt.xlabel("Time")
plt.ylabel("Y")
plt.show()
