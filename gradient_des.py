from typing import List
from tools import gradient_descent, norm, backtracking

def max_descent(tol:float, max_iter:int, x0:int, y0:int)->List[int]:
    k:int = 0
    solution:List[float] = [x0, y0]
    while k <= max_iter:
        pk = gradient_descent(x0, y0)
        if(norm(pk)<tol):
            return solution
        alpha = backtracking()
        solution = solution + (alpha * pk)

        k+=1

    return solution