import numpy as np
import sys
from numpy.linalg import norm
np.seterr(all='warn')

def f(X):
    x = X[0]
    y = X[1]
    v= x**2+y**2
    return v

# Производная (градиент) функции сферы
def df(X):
    v = np.copy(X)
    v[0] = X[0]*2
    v[1] = X[1]*2
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
    v = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
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
    v[0] = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
    v[1] = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)

    return v


# F_ROSENBROCK is a Rosenbrock function
# 	v = F_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value

def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x)**2 + 100*(y - x**2)**2
    return v

# DF_ROSENBROCK is a Rosenbrock function derivative
# 	v = DF_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfR(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = -2 * (1 - x) + 200 * (y - x**2)*(- 2 * x)
    v[1] = 200 * (y - x**2)
    return v



def H(x0, tol, df):
    n = len(x0)
    deltaX = 0.1 * tol
    Hessian = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x1 = np.array(x0, dtype=float)
            x2 = np.array(x0, dtype=float)

            if i == j:
                x1[i] += deltaX
                x2[i] -= deltaX
                Hessian[i, j] = (df(x1) - df(x2))[i] / (deltaX * 2)
            else:
                x1[i] += deltaX
                x2[i] -= deltaX

                dx1 = df(x1)
                dx2 = df(x2)

                Hessian[i, j] = (dx1 - dx2)[j] / (deltaX * 2)

    return Hessian


def nsearch(f, df, x0, tol):
# NSEARCH searches for minimum using Newton method
# 	answer_ = nsearch(f, df, x0, tol)
#   INPUT ARGUMENTS
#   f  - objective function
#   df - gradient
# 	x0 - start point
# 	tol - set for bot range and function value
#   OUTPUT ARGUMENTS
#   answer_ = [xmin, fmin, neval, coords]
# 	xmin is a function minimizer
# 	fmin = f(xmin)
# 	neval - number of function evaluations
#   coords - array of statistics

    kmax =1000
    k=0
    coords =[x0]
    while (True):
        g = df(x0)
        H0 = H(x0,tol,df)
        dx = np.linalg.lstsq(-H0,g)[0]
        x0 = x0 + dx
        coords.append(x0)
        if ((norm(dx) < tol) or (k >= kmax)):
            answer_ = [x0, f(x0), k,  coords]
            return answer_
        k += 1
print(nsearch(f,df,[0,0],0.001))


