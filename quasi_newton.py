import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools import extend_list, fxy
from tools import norm, beale_function_vectorized, beale_function_gradient_vectorized, backtracking_line_search
from animation import animar_puntos

def bfgs(tol: float, max_iter:int, x0:float, y0: float, B):
    k:int = 0

    solution = np.array([x0, y0])

    solutions = [solution]
    x_s = [x0]
    y_s = [y0]

    ks = []
    gradients = []
    norms = []
    pks = []
    alphas = []
    sks = []
    yks = []
    Bs = []

    while k <= max_iter:
        gradients.append(beale_function_gradient_vectorized(solution))
        p_k = -B @ beale_function_gradient_vectorized(solution)

        if norm(p_k) < tol:
            break

        alpha = backtracking_line_search(beale_function_vectorized, beale_function_gradient_vectorized, solution, p_k)

        new_sol = solution + (alpha * p_k)
        solution = new_sol

        x_s.append(solution[0])
        y_s.append(solution[1])

        solutions.append(solution)
        s_k = solutions[-1] - solutions[-2]
        y_k = beale_function_gradient_vectorized(solutions[-1]) - beale_function_gradient_vectorized(solutions[-2])
        B = cuasi_update_BFGS(B, s_k, y_k)

        ks.append(k)
        pks.append(p_k)
        norms.append(norm(beale_function_gradient_vectorized(solution)))
        alphas.append(alpha)
        sks.append(s_k)
        yks.append(yks)
        Bs.append(B)

        k += 1

    plotting(x_s, y_s)
    pts = solutions
    max_length = max(len(ks), len(solutions), len(gradients), len(pks), len(norms), len(alphas), len(sks), len(yks), len(Bs))

    ks = extend_list(ks, max_length, 0)
    gradients = extend_list(gradients, max_length, 0)
    solutions = extend_list(solutions, max_length, 0)
    pks = extend_list(pks, max_length, 0)
    norms = extend_list(norms, max_length, 0)
    alphas = extend_list(alphas, max_length, 0)
    sks = extend_list(sks, max_length, 0)
    yks = extend_list(yks, max_length, 0)
    Bs = extend_list(Bs, max_length, 0)

    data = {
        'k' : ks,
        'x_k': solutions,
        '||∇f(x)||': norms,
        'p_k': pks,
        'alpha': alphas,
        'B_k+1': Bs
    }
    animar_puntos(pts)
    df = pd.DataFrame(data)
    return df

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