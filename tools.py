import numpy as np
from typing import List

def fxy(x, y):
    term1 = (1.5 - x + (x*y)) ** (2)
    term2 = (2.25 - x + (x*(y**2))) ** (2)
    term3 = (2.625 - x + (x*(y**3))) ** (2)

    return term1 + term2 + term3

def gradient(x, y):
    dfdx = (2*x*(y**6)) + (2*x*(y**4)) + (5.25*(y**3)) - (4*x*(y**3)) + (4.5*(y**2)) - (2*x*(y**2)) + (3*y) - (4*x*y) + (6*x) - 12.75
    dfdy = (6*(x**2)*(y**5)) + (4*(x**2)*(y**3)) - (6*(x**2)*(y**2)) - (2*y*(x**2)) - (2*(x**2)) + (15.75*x*(y**2)) + (9*x*y) + (3*x)

    return [dfdx, dfdy]

def gradient_descent(x:float, y:float)->List[float]:
    dfdx = (-2*x*(y**6)) - (2*x*(y**4)) - (5.25*(y**3)) + (4*x*(y**3)) - (4.5*(y**2)) + (2*x*(y**2)) - (3*y) + (4*x*y) - (6*x) + 12.75
    dfdy = (-6*(x**2)*(y**5)) - (4*(x**2)*(y**3)) + (6*(x**2)*(y**2)) + (2*y*(x**2)) + (2*(x**2)) - (15.75*x*(y**2)) - (9*x*y) - (3*x)

    return [dfdx, dfdy]


def hessian(x, y):
    dfdxx = (2*(y**6)) + (2*(y**4)) - (4*(y**3)) - (2*(y**2)) - (4*y) + 6
    dfdxy = (12*x*(y**5)) + (8*x*(y**3)) + (15.75*(y**2)) - (12*x*(y**2)) + (9*y) - (4*x*y) + 3 - (4*x)

    dfdyx = (12*x*(y**5)) + (8*x*(y**3)) - (12*x*(y**2)) - (4*x*y) - (4*x) + (15.75*(y**2)) + (9*y) + 3
    dfdyy = (30*(x**2)*(y**4)) + (12*(x**2)*(y**2)) - (12*(x**2)*y) - (2*(x**2)) + (31.5*x*y) + (9*x)

    return [[dfdxx, dfdxy], [dfdyx, dfdyy]]

def norm(xvec:List[float])->float:
    c = 0
    for cord in xvec:
        c += (cord**2)
    return (c**0.5)

def backtracking(alpha:float, p:float=0.5, u:float):
    while phialpha(alpha) > (phialpha(0) + (u*alpha*dphialpha(0))):
        alpha = (p*alpha)

    return alpha

def phialpha()