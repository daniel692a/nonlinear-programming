import numpy as np
import matplotlib.pyplot as plt
from tools import gradient, fxy
from tools import norm, beale_function_vectorized, beale_function_gradient_vectorized, backtracking_line_search

def bfgs(tol: float, max_iter:int, x0:float, y0: float, B):
    k:int = 0

    solution = np.array([x0, y0])

    solutions = [solution]
    x_s = [x0]
    y_s = [y0]

    while k <= max_iter:
        p_k = -B @ beale_function_gradient_vectorized(solution)

        if norm(p_k) < tol:
            break

        alpha = backtracking_line_search(beale_function_vectorized, beale_function_gradient_vectorized, solution, p_k)

        new_sol = solution + (alpha * p_k)
        solution = new_sol

        x_s.append(solution[0])
        y_s.append(solution[1])
        print(f'nueva solución: {solution}\n')
        solutions.append(solution)
        s_k = solutions[-1] - solutions[-2]
        y_k = beale_function_gradient_vectorized(solutions[-1]) - beale_function_gradient_vectorized(solutions[-2])
        B = cuasi_update_BFGS(B, s_k, y_k)

        k += 1

    plotting(x_s, y_s)
    return solution

def cuasi_update_BFGS(B_k, s_k, y_k):
    Bs = B_k @ y_k
    yBs = y_k.T @ Bs
    syk = s_k.T @ y_k
    term1 = (1 + (yBs / syk)) * np.outer(s_k, s_k) / syk
    term2 = (np.outer(s_k, Bs.T) + np.outer(Bs, s_k.T)) / syk
    B_k1 = B_k + term1 - term2
    return B_k1


def plotting(x_sol:np.ndarray, y_sol:np.ndarray):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = fxy(X, Y)
    plt.figure()
    contour = plt.contour(X, Y, Z)
    plt.plot(x_sol, y_sol)
    plt.title('Beale Function')
    plt.clabel(contour, inline=True, fontsize=8)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(x_sol, y_sol, 'ro-', label='Soluciones en búsqueda en línea')
    plt.legend()
    plt.show()