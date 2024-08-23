import numpy as np
import sys
from numpy.linalg import norm

#F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def f(xx):


    x1 = xx[0]
    x2 = xx[1]

    term1 = (x1 + 2*x2 - 7)**2
    term2 = (2*x1 + x2 - 5)**2

    y = term1 + term2

    return y

def df(xx):
    x1 = xx[0]
    x2 = xx[1]
    v = np.array([0, 0], dtype=float)
    v[0] = 10*x1 + 8*x2 - 34
    v[1] = 8*x1 + 10*x2 - 38

    return v


def grsearch(x0,tol):

# GRSEARCH searches for minimum using gradient descent method
# 	answer_ = grsearch(x0,tol)
#   INPUT ARGUMENTS
#	x0 - starting point
# 	tol - set for bot range and function value
#   OUTPUT ARGUMENTS
#   answer_ = [xmin, fmin, neval, coords]
# 	xmin is a function minimizer
# 	fmin = f(xmin)
# 	neval - number of function evaluations
#   coords - array of x values found during optimization    

    al = 0.01
    kmax = 1000
    k=1
    coords = [x0]
    while (True):
        g = -df(np.array(x0))
        x1 = x0 + al * g
        deltaX = x1 - x0
        if ((norm(deltaX) < tol) or (k >= kmax)):
            xmin = x1
            answer_ = [xmin, f(xmin), k, coords]
            return answer_
        x0 = x1
        coords.append(x1)
        k+=1

print(grsearch([1,1],0.0001))
