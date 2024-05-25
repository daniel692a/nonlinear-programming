from typing import List
import numpy as np
import matplotlib.pyplot as plt
from tools import gradient_descent, norm, backtracking, fxy

def max_descent(tol:float, max_iter:int, x0:float, y0:float)->List[int]:
    k:int = 0
    solution = np.array([x0, y0])

    x_s = [x0]
    y_s = [y0]

    a = float(input("Ingrese el alpha incial: "))
    p = float(input('Ingrese el valor p: '))
    u = float(input('Ingrese el valor de u: '))

    while k <= max_iter:
        pk = gradient_descent(solution[0], solution[1])

        print(f'dirección de descenso: {pk}')

        if(norm(pk)<tol):
            return solution

        alpha = backtracking(a, p, u, solution[0], solution[1], pk)
        solution = solution + (alpha * pk)

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