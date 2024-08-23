import numpy as np
import sys
from numpy.linalg import norm
from numpy.linalg import inv
np.seterr(divide='ignore', invalid='ignore')

def fmat(xx):
    x1 = xx[0]
    x2 = xx[1]
    term1 = 0.26 * (x1**2 + x2**2)
    term2 = -0.48 * x1 * x2
    y = term1 + term2
    return y

def dfmat(xx):


    x1 = xx[0]
    x2 = xx[1]
    v = np.copy(xx)
    v[0] = 0.52 * x1 - 0.48 * x2
    v[1] = 0.52 * x2 - 0.48 * x1

    return v
# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fH(X):
    x = X[0]
    y = X[1]
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return v


# DF_HIMMELBLAU is a Himmelblau function derivative
# 	v = DF_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfH(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)

    return v


def nagsearch(f, df, x0, tol):
    al = 0.05
    eta = al / 10
    gamma = 0.75
    
    xk = np.copy(x0)
    yk = np.copy(x0)
    neval = 0
    coords = [xk]
    k = 0
    
    while True:
        grad_yk = df(yk)
        neval += 1

        xk_new = yk - eta * grad_yk
        yk_new = xk_new + gamma * (xk_new - xk)
        
        if norm(grad_yk) < tol or k >= 1000:
            xmin = xk_new
            fmin = f(xmin)
            answer_ = [xmin, fmin, neval, coords]
            return answer_
        
        xk = xk_new
        yk = yk_new
        k += 1
        coords.append(xk)


print(nagsearch(fH,dfH,(1,1), 10e-3))
