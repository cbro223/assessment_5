# import statements
import numpy as np
import matplotlib.pyplot as plt


def difference_squares(x, y):
    z_1 = x ** 2 - y ** 2
    z_2 = (x - y) * (x + y)
    print("z1 = {:.32f}".format(z_1))
    print("z2 = {:.32f}".format(z_2))
    print("z2 = {:.32f}".format(z_2 - z_1))
    pass


def relative_error_subtraction(x, y, z_exact):
    print("x = {:.64f}".format(x))
    print("y = {:.64f}".format(y))
    z_approx = x - y
    print("approx. value of z = {:.64f}".format(z_approx))
    print("exact value of z = {:.64f}".format(z_exact))
    rel_error = abs(z_exact - z_approx) / abs(z_exact)
    print("relative error = {:.16f}".format(rel_error))
    pass


def exact_solution_ode1(t):
    term_1 = 2 * np.multiply(t, np.exp(np.multiply(-5, t)))
    term_2 = 4 * np.exp(np.multiply(-5, t))
    output = term_1 + term_2
    return np.transpose(output)
    pass


def mean_absolute_error(y_exact, y_approx):
    diff = 0
    for i in range(1, len(y_exact)):
        diff = diff + abs(y_exact[i] - y_approx[0][i])
    output = diff / len(y_exact)
    print(output)
    return output
    pass


def derivative_ode1(t, y):
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
    """ TODO
    """
    pass


def explicit_rk_solver(f, tspan, y0, h, alpha, beta, gamma):
    """ TODO
    """
    pass
