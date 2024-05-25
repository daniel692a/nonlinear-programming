import numpy as np
import matplotlib.pyplot as plt
from tools import hessian, gradient, norm, backtracking, is_positive_definite, fxy

def newton_des(tol:float, max_iter:int, x0:float, y0:float)->np.ndarray:
    k:int = 0

    solution = np.array([x0, y0])

    x_s = [x0]
    y_s = [y0]

    a = float(input("Ingrese el alpha incial: "))
    p = float(input('Ingrese el valor p: '))
    u = float(input('Ingrese el valor de u: '))

    while k <= max_iter:

        hess_f = hessian(solution[0], solution[1])

        p_k = ( -np.linalg.inv(hess_f) ) @ gradient(solution[0], solution[1])

        if(norm(gradient(solution[0], solution[1])) < tol):
            break

        if(is_positive_definite(hess_f)):
            alpha = 1
        else:
            alpha = backtracking(a, p, u, solution[0], solution[1], p_k)

        solution = solution + (alpha * p_k)

        x_s.append(solution[0])
        y_s.append(solution[1])
        print(f'nueva solución: {solution}\n')

        k += 1

    plotting(x_s, y_s)

    return solution


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
