import numpy as np
from typing import List

def fxy(x:float, y:float)->float:
    term1 = (1.5 - x + (x*y)) ** (2)
    term2 = (2.25 - x + (x*(y**2))) ** (2)
    term3 = (2.625 - x + (x*(y**3))) ** (2)

    return term1 + term2 + term3


def gradient(x:float, y:float)->np.ndarray:
    dfdx = (2*x*(y**6)) + (2*x*(y**4)) + (5.25*(y**3)) - (4*x*(y**3)) + (4.5*(y**2)) - (2*x*(y**2)) + (3*y) - (4*x*y) + (6*x) - 12.75
    dfdy = (6*(x**2)*(y**5)) + (4*(x**2)*(y**3)) - (6*(x**2)*(y**2)) - (2*y*(x**2)) - (2*(x**2)) + (15.75*x*(y**2)) + (9*x*y) + (3*x)

    return np.array([dfdx, dfdy])


def gradient_descent(x:float, y:float)->np.ndarray:
    dfdx = (-2*x*(y**6)) - (2*x*(y**4)) - (5.25*(y**3)) + (4*x*(y**3)) - (4.5*(y**2)) + (2*x*(y**2)) - (3*y) + (4*x*y) - (6*x) + 12.75
    dfdy = (-6*(x**2)*(y**5)) - (4*(x**2)*(y**3)) + (6*(x**2)*(y**2)) + (2*y*(x**2)) + (2*(x**2)) - (15.75*x*(y**2)) - (9*x*y) - (3*x)

    return np.array([dfdx, dfdy])



def hessian(x:float, y:float)->List[List[float]]:
    dfdxx = (2*(y**6)) + (2*(y**4)) - (4*(y**3)) - (2*(y**2)) - (4*y) + 6
    dfdxy = (12*x*(y**5)) + (8*x*(y**3)) + (15.75*(y**2)) - (12*x*(y**2)) + (9*y) - (4*x*y) + 3 - (4*x)

    dfdyx = (12*x*(y**5)) + (8*x*(y**3)) - (12*x*(y**2)) - (4*x*y) - (4*x) + (15.75*(y**2)) + (9*y) + 3
    dfdyy = (30*(x**2)*(y**4)) + (12*(x**2)*(y**2)) - (12*(x**2)*y) - (2*(x**2)) + (31.5*x*y) + (9*x)

    return np.array([[dfdxx, dfdxy], [dfdyx, dfdyy]])


def norm(xvec:np.ndarray)->float:
    return np.linalg.norm(xvec)

def backtracking(alpha:float, p:float, u:float, x:float, y:float, p_k:np.ndarray)->float:
    while fxy(x+(alpha*p_k[0]), y+(alpha*p_k[0])) > ( fxy(x, y) + (u*alpha*(np.dot(gradient(x, y), p_k) ))):
        alpha = (p*alpha)
    return alpha


def is_positive_definite(matrix: np.ndarray) -> bool:
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues > 0)