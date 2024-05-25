import numpy as np
from tools import gradient, fxy, backtracking

def bfgs(tol: float, max_iter:int, x0:float, y0: float, B:np.ndarray[np.ndarray[float]])->np.ndarray[float]:
    k:int = 0

    solution:np.ndarray[float] = np.array([x0, y0])

    while k <= max_iter:
        p_k = -B * gradient(solution[0], solution[1])

        alpha = backtracking(a, p, u, solution[0], solution[1], p_k)

        new_sol = solution + (alpha * p_k)
        s_k = new_sol - solution
        y_k = gradient(new_sol[0], new_sol[1]) - gradient(solution[0], solution[1])

        B = cuasi_update_BFGS()

        k += 1

    return solution

def cuasi_update_BFGS(B:np.ndarray[np.ndarray[float]], s_k:np.ndarray[float], y_k:np.ndarray[float])->np.ndarray[np.ndarray[float]]:
    term1  = 1 + ( ( y_k @ B @ np.transpose(y_k) ) / ( s_k @ np.transpose(y_k)) )
    term2 = ( np.transpose(s_k) @ s_k ) / ( s_k @ np.transpose(y_k) )
    term3 = ( np.transpose(s_k) @ y_k @ B ) / ( s_k @ np.transpose(y_k) )
    term4 = ( B @ np.transpose(y_k) @ s_k ) / ( s_k @ np.transpose(y_k) )

    return B + (term1*term2) - term3 - term4