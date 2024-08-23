import numpy as np
import sys
from numpy.linalg import norm


def goldensectionsearch(f, interval, tol):
    neval = 2
    fi = (1+np.sqrt(5))/2
    [a,b] = interval 
    coords=[]
    x1 = b - (b-a)/fi
    x2 = a + (b-a)/fi
    fx1 = f(x1)
    fx2 = f(x2)
    while (np.abs(b-a) > tol):
        if (fx1 > fx2):
            a = x1
            x1 = x2
            fx1 = fx2
            x2 = a + (b-a)/fi
            fx2 = f(x2)
        else:
            b = x2
            x2 = x1
            fx2 = fx1
            x1 = b - (b-a)/fi
            fx1 = f(x1)
        coords.append([x1,x2,a,b])
        neval +=2
    xmin = (a+b)/2

    answer_ = [xmin, f(xmin), neval]
    return answer_



# F_ROSENBROCK is a Rosenbrock function
# 	v = F_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
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



def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
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
    v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)
    v[1] = 200 * (y - x ** 2)
    return v


def bbsearch(f, df, x0, tol):

# BBSEARCH searches for minimum using stabilized BB1 method
# 	answer_ = bbsearch(f, df, x0, tol)
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

    kmax = 1000
    k=1
    D = 0.1
    coords = []
    coords.append(x0)
    g0 = df(x0)

    f1dim = lambda alpha:f(x0 + alpha*g0)
    al = goldensectionsearch(f1dim,[0,1],tol)[0]

    while (True):

        x1 = x0 - al * g0
        g1 = df(x1)
        deltaG = g1 - g0
        deltaX = x1 - x0

        al = (np.dot(deltaX.T, deltaX)/np.dot(deltaX.T , deltaG))
        al_stab = D/(norm(g1))

        if al_stab < al: 
            al = al_stab

        if ((norm(deltaX) < tol) or (k >= kmax)):
            xmin = x1
            answer_ = [xmin, f(xmin), k, coords]
            return answer_
        g0 = g1
        x0 = x1

        coords.append(x1)
        k+=1
print(bbsearch(f,df, [1,1], 10e-5))