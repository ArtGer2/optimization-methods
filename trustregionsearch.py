import numpy as np
import sys
from numpy.linalg import norm
from numpy.linalg import inv



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


# F_ROSENBROCK is a Rosenbrock function
# 	v = F_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value

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


def goldensectionsearch(f, interval, tol):
    a = interval[0]
    b = interval[1]
    Phi = (1 + np.sqrt(5)) / 2
    L = b - a
    x1 = b - L / Phi
    x2 = a + L / Phi
    y1 = f(x1)
    y2 = f(x2)
    neval = 2
    xmin = x1
    fmin = y1
    # main loop
    while np.abs(L) > tol:
        if y1 > y2:
            a = x1
            xmin = x2
            fmin = y2
            x1 = x2
            y1 = y2
            L = b - a
            x2 = a + L / Phi
            y2 = f(x2)
            neval += 1
        else:
            b = x2
            xmin = x1
            fmin = y1
            x2 = x1
            y2 = y1
            L = b - a
            x1 = b - L / Phi
            y1 = f(x1)
            neval += 1

    answer_ = [xmin, fmin, neval]
    return answer_


def pparam(pU, pB, tau):
    if (tau <= 1):
        p = np.dot(tau, pU)
    else:
        p = pU + (tau - 1) * (pB - pU)
    return p


def doglegsearch(mod, g0, B0, Delta, tol):
    # dogleg local search
    xcv = np.dot(-g0.transpose(), g0) / np.dot(np.dot(g0.transpose(), B0), g0)
    pU = xcv *g0
    xcvb = inv(- B0)
    pB = np.dot(inv(- B0), g0)

    func = lambda x: mod(np.dot(x, pB))
    al = goldensectionsearch(func, [-Delta / norm(pB), Delta / norm(pB)], tol)[0]
    pB = al * pB
    func_pau = lambda x: mod(pparam(pU, pB, x))
    tau = goldensectionsearch(func_pau, [0, 2], tol)[0]
    pmin = pparam(pU, pB, tau)
    if norm(pmin) > Delta:
        pmin_dop = (Delta / norm(pmin))
        pmin = np.dot(pmin_dop, pmin)
    return pmin




def trustreg(f, df, x0, tol):
    coordinates = [x0]
    radii = []
    eta = 0.1
    Delta = 1 / 4
    dmax = 0.1
    radii.append(Delta)
    B = np.eye(len(x0))
    H0 = B
    kmax = 1000
    dk = np.ones_like(x0)

    neval = 0

    while norm(dk) >= tol and neval < kmax:
        mk = lambda x: (f(x0) + np.dot(np.array(x).transpose(), df(x0)) + 1 / 2 * np.dot(np.dot(np.array(x).transpose(), B),np.array(x)))
        pk = doglegsearch(mk, df(x0), B, Delta, tol)
        ro = (f(x0) - f(x0 + pk)) / (mk(0) - mk(pk))[0][0]

        if ro > eta:
            xmin = x0 + pk
        else:
            xmin = x0

        if ro < 0.25:
            Delta *= 0.25
        elif ro > 0.75 and norm(pk) == Delta:
            Delta = min(2 * Delta, dmax)

        radii.append(Delta)
        coordinates.append(xmin)
        dk = xmin - x0
        yk = df(xmin) - df(x0)
        if (norm(dk)!=0):
            
            H = H0 + np.dot(dk, dk.transpose()) / np.dot(dk.transpose(), yk) - np.dot(H0, np.dot(np.dot(yk, yk.transpose()), H0)) / np.dot(np.dot(yk.transpose(), H0), yk)
            B = np.linalg.inv(H)
        else:
            dk = 100
        H0 = H
        x0 = xmin
        neval += 1

    return xmin, f(xmin), neval, coordinates, radii


#print(trustreg(fR,dfR,(-1,-1),10e-3))
