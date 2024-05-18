# import statements
import numpy as np
import matplotlib.pyplot as plt


def difference_squares(x, y):
    """
    Will compute the difference between of squares by two different methods and prints the difference of these two methods of computation
    Parameters:
        x(float) - the variable x
        y (float) - the variable y
    Returns:
        none
    Examples
        if x = 1, y = 2, will compute the floating point difference between 1**2-2**2 and (1-2)(1+2)
    """
    z_1 = x ** 2 - y ** 2
    z_2 = (x - y) * (x + y)
    print("z1 = {:.32f}".format(z_1))
    print("z2 = {:.32f}".format(z_2))
    print("difference = {:.32f}".format(z_2 - z_1))
    pass


def relative_error_subtraction(x, y, z_exact):
    """
    This function computes the relative error of the exact difference and the computed difference between two numbers
    Parameters:
        x (float)- the variable x of x-y
        y (float)- the variable y of x-y
        z_exact (float)- the exact difference between the two numbers
    Returns: None
    """
    print("x = {:.64f}".format(x))
    print("y = {:.64f}".format(y))
    z_approx = x - y
    print("approx. value of z = {:.64f}".format(z_approx))
    print("exact value of z = {:.64f}".format(z_exact))
    rel_error = abs(z_exact - z_approx) / abs(z_exact)
    print("relative error = {:.16f}".format(rel_error))
    pass


def exact_solution_ode1(t):
    """
    The hand solved exact solution of the ODE problem.
    Parameters
        t (float array) - the time values which are to be calculated for the exact solution of y
    returns:
        y (float array) the exact values of y for the given t values
    """
    term_1 = 2 * np.multiply(t, np.exp(np.multiply(-5, t)))
    term_2 = 4 * np.exp(np.multiply(-5, t))
    output = term_1 + term_2
    return np.transpose(output)
    pass


def mean_absolute_error(y_exact, y_approx):
    """
    Computes the mean absolute value of exact versus approximate values for an ODE
    Parameters:
        y_exact (float array) - an array of the exact y values
        y_approx (float array) - an array of the approximate/numerical solved y values
    Returns:
        mae (float) - the mean absolute error of the exact versus approximate values
    Notes:
        (precondition 1): the y_exact and y_approx arrays should exclude the first time value(i.e. the boundary condition)
    """
    diff = 0
    for i in range(len(y_exact)):
        diff = diff + abs(y_exact[i] - y_approx[i])
    output = diff / len(y_exact)
    return output
    pass


def derivative_ode1(t, y):
    """
    This function outputs the result of the ODE of -5y + 2e^(-5t)
    Parameters:
        t (float) - the time value to evaluate the ODE at
        y (float) - the y value to evaluate the ODE at
    Returns:
        dy/dt (float) - the derivative value at the provided t and y values
    """
    return -5 * y + 2 * np.exp(-5 * t)
    pass


def euler_step(f, t, y, h):
    """
    Calculate one step of the Euler method.

    Parameters
    ----------
    f : function
        Derivative function (callable).
    t : float
        Independent variable at start of step.
    y : float
        Dependent variable at start of step.
    h : float
        Step size along independent variable.

    Returns
    -------
    y_new : float
        Dependent variable at end of step.
    """
    f0 = f(t, y)
    y_new = y + h * f0
    return y_new


def improved_euler_step(f, t, y, h):
    """
    Calculate one step of the Improved Euler method.

    Parameters
    ----------
    f : function
        Derivative function (callable).
    t : float
        Independent variable at start of step.
    y : float
        Dependent variable at start of step.
    h : float
        Step size along independent variable.

    Returns
    -------
    y_new : float
        Dependent variable at end of step.
    """
    f0 = f(t, y)
    f1 = f(t + h, y + h * f0)
    y_new = y + h * 0.5 * (f0 + f1)
    return y_new


def classic_rk4_step(f, t, y, h):
    """
    Calculate one step of the Classic RK4 method.

    Parameters
    ----------
    f : function
        Derivative function (callable).
    t : float
        Independent variable at start of step.
    y : float
        Dependent variable at start of step.
    h : float
        Step size along independent variable.

    Returns
    -------
    y_new : float
        Dependent variable at end of step.
    """
    f0 = f(t, y)
    f1 = f(t + h * 0.5, y + h * 0.5 * f0)
    f2 = f(t + h * 0.5, y + h * 0.5 * f1)
    f3 = f(t + h, y + h * f2)
    y_new = y + h * (f0 + 2. * f1 + 2. * f2 + f3) / 6.
    return y_new


def explicit_rk_step(f, t, y, h, alpha, beta, gamma):
    """
    This function computes one step forward for an ODE using a provided Runge-Kutta method
    Parameters:
        f (function) - the ODE function used for slope
        t (float) - the current time value
        y (float) - the current function value
        h (float) - the desired step size
        alpha (array) - the alpha matrix for the RK method
        beta (array) - the beta matrix for the RK method
        gamma (matrix) - the gamma matrix for the RK method
    Returns:
        y_new (float) - The next value of the function evaluated at t + h
    Notes:
        (precondition 1): alpha and beta must be n rows
        (precondition 2): gamma matrix must be n x n in size
        (precondition 3): the step size must be positive
    """
    # Check that the preconditions are met 
    if len(alpha) != len(beta) or len(gamma) != len(alpha):
        raise ValueError("Input matrices need to be the same lengths")
    derivative_evaluations = np.zeros(len(alpha))
    # Find all the derivative evalautions from f0 to fn

    for i in range(len(beta)):
        # Cycle through the beta rows
        t_step = t + h * beta[i]
        y_step = y
        # Cycle through the columns of the gamma matrix
        for j in range(len(beta)):
            y_step += h * (gamma[i][j] * derivative_evaluations[j])
        derivative_evaluations[i] = f(t_step, y_step)

    # Generate the next step distance
    step_distance = 0
    for i in range(len(derivative_evaluations)):
        step_distance += h * alpha[i] * derivative_evaluations[i]

    # Compute the next y value iwth the previous step distance
    y_new = step_distance + y
    return y_new


def explicit_rk_solver(f, tspan, y0, h, alpha, beta, gamma):
    """
    This function estimates the function values over a time span using a provided Runge-Kutta method
    Parameters:
        f (function) - the derivative function/ODE
        tspan (array) - the starting time and desired ending time
        y0 (float) - the initial value of the function (i.e. the boundary condition)
        h (float) - the desired step size of the RK method
        alpha (array) - the alpha matrix for the RK method
        beta (array) - the beta matrix for the RK method
        gamma (matrix) - the gamma matrix for the RK method
    Returns:
        t (array) - an array of all the times the function was estimated at
        y (array) - an array of all the estimated y values
    Notes:
        (precondition 1): alpha and beta must be n rows
        (precondition 2): gamma matrix must be n x n in size
        (precondition 3): the step size must be positive
    """
    # Check that the preconditions are met 
    if len(alpha) != len(beta) or len(gamma) != len(alpha):
        raise ValueError("Input matrices need to be the same lengths")

    t = np.arange(tspan[0], tspan[1] + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = explicit_rk_step(f, t[i-1], y[i-1], h, alpha, beta, gamma)
    print(len(y))
    return t,y
    pass
