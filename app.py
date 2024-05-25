import numpy as np
from gradient_des import max_descent
from newton import newton_des

if __name__ == '__main__':
    tol = float(input("Ingrese la tolerancia de error: "))
    max_iter = int(input("Ingrese el número máximo de iteraciones: "))
    x0 = float(input('Ingrese la coordenada x inicial: '))
    y0 = float(input('Ingrese la coordenada y inicial: '))
    ans = int(input('Opciones de métodos\n1.Gradiente\n2.Newton\n'))

    if ans == 1:
        min = max_descent(tol, max_iter, x0, y0)
    elif ans == 2:
        min = newton_des(tol, max_iter, x0, y0)

    print(f'Mínimo encontrado en: {min}')


