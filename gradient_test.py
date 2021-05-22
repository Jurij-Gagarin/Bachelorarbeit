from scipy import optimize as opt
import numpy as np
import time


def function(x, a):
    return (a[0] - x[0])**2 + a[1]*(x[1] - x[0]**2)**2


def jacobian_function(x, a):
    return np.array([-2*(a[0]-x[0])-4*a[1]*x[0]*(x[1]-x[0]**2), 2*a[1]*(x[1]-x[0]**2)])


def minimize_function():
    x0 = np.zeros(2)
    return opt.minimize(function, x0, jac=jacobian_function, args=([1.e-07, 10]), tol=1.e-15)


def derivative(i, x, y, k):
    return .5*k*()




print(minimize_function())
