import numpy as np
from typing import List

def fxy(x:float, y:float)->float:
    term1 = (1.5 - x + x * y) ** 2
    term2 = (2.25 - x + x * y ** 2) ** 2
    term3 = (2.625 - x + x * y ** 3) ** 2

    return term1 + term2 + term3

def gradient(x:float, y:float)->np.ndarray:
    dfdx = 2 * (1.5 - x + x * y) * (-1 + y) + 2 * (2.25 - x + x * y ** 2) * (-1 + y ** 2) + 2 * (2.625 - x + x * y ** 3) * (-1 + y ** 3)
    dfdy = 2 * (1.5 - x + x * y) * x + 2 * (2.25 - x + x * y ** 2) * 2 * x * y + 2 * (2.625 - x + x * y ** 3) * 3 * x * y ** 2
    return np.array([dfdx, dfdy])


def hessian(x:float, y:float)->List[List[float]]:
    dfdxx = 2 * (-1 + y) ** 2 + 2 * (-1 + y ** 2) ** 2 + 2 * (-1 + y ** 3) ** 2
    dfdyy = 2 * x ** 2 + 8 * x ** 2 * y ** 2 + 18 * x ** 2 * y ** 4
    dfdxy = 2 * (-1 + y) * x + 4 * (-1 + y ** 2) * x * y + 6 * (-1 + y ** 3) * x * y ** 2

    return np.array([[dfdxx, dfdxy], [dfdxy, dfdyy]])

def beale_function_vectorized(v):
    return fxy(v[0], v[1])

def beale_function_gradient_vectorized(v):
    return gradient(v[0], v[1])


def norm(xvec:np.ndarray)->float:
    sum_of_squares = sum(component ** 2 for component in xvec)
    return np.sqrt(sum_of_squares)

def backtracking(alpha:float, p:float, u:float, x:float, y:float, p_k:np.ndarray)->float:
    while fxy(x+(alpha*p_k[0]), y+(alpha*p_k[0])) > ( fxy(x, y) + (u*alpha*(np.dot(gradient(x, y), p_k) ))):
        alpha = (p*alpha)
    return alpha

def backtracking_line_search(f, grad_f, x, p, alpha=1, rho=0.5, c=1e-4):
    while f(x + alpha * p) > f(x) + c * alpha * np.dot(grad_f(x), p):
        alpha *= rho
    return alpha


def is_positive_definite(matrix: np.ndarray) -> bool:
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues > 0)