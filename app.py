import numpy as np
from tools import beale_function_vectorized
from gradient_des import max_descent
from newton import newton_des
from quasi_newton import bfgs
from gui_gradient import gui_table

if __name__ == '__main__':
    print('Matemáticas Avanzadas para la Ciencia de Datos')
    print('Daniel Armas Ramírez')
    print('Función de Beale')
    while True:

        tol = float(input("Ingrese la tolerancia de error: "))
        max_iter = int(input("Ingrese el número máximo de iteraciones: "))
        x0 = float(input('Ingrese la coordenada x inicial: '))
        y0 = float(input('Ingrese la coordenada y inicial: '))

        min = max_descent(tol, max_iter, x0, y0)
        print(f'Mínimo encontrado con Gradiente descendiente en: {min}\nValor de fxy: {beale_function_vectorized(min)}')
        min = newton_des(tol, max_iter, x0, y0)
        print(f'Mínimo encontrado con Newton en: {min}\nValor de fxy: {beale_function_vectorized(min)}')
        min = bfgs(tol, max_iter, x0, y0, np.identity(2))
        gui_table(min, 'Quasi-Newton')

        break


