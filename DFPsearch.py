import numpy as np
import sys
from numpy.linalg import norm
#np.seterr(divide='ignore', invalid='ignore')


def fSphere(X):
    return np.sum(X**2)

def dfSphere(X):
    return 2 * X

def fLevy(X):
    w = 1 + (X - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = (w[0] - 1)**2 * (1 + 10 * np.sin(np.pi * w[0] + 1)**2)
    term3 = (w[1] - 1)**2 * (1 + np.sin(2 * np.pi * w[1])**2)
    return term1 + term2 + term3

def dfLevy(X):
    w = 1 + (X - 1) / 4
    dfdw = np.zeros_like(X)
    dfdw[0] = 2 * np.sin(np.pi * w[0]) * np.pi * np.cos(np.pi * w[0]) + \
              2 * (w[0] - 1) * (1 + 10 * np.sin(np.pi * w[0] + 1)**2) + \
              20 * (w[0] - 1)**2 * np.sin(np.pi * w[0] + 1) * np.pi * np.cos(np.pi * w[0] + 1)
    dfdw[1] = 2 * (w[1] - 1) * (1 + np.sin(2 * np.pi * w[1])**2) + \
              2 * (w[1] - 1)**2 * np.sin(2 * np.pi * w[1]) * 2 * np.pi * np.cos(2 * np.pi * w[1])
    df = dfdw * 1/4
    return df


def zoom(phi, dphi, alo, ahi, c1, c2):
    j = 1
    jmax = 1000
    while j < jmax:
        a = cinterp(phi, dphi, alo, ahi)
        if phi(a) > phi(0) + c1 * a * dphi(0) or phi(a) >= phi(alo):
            ahi = a
        else:
            if abs(dphi(a)) <= -c2 * dphi(0):
                return a  # a is found
            if dphi(a) * (ahi - alo) >= 0:
                ahi = alo
            alo = a
        j += 1
    return a


def cinterp(phi, dphi, a0, a1):
    if np.isnan(dphi(a0) + dphi(a1) - 3 * (phi(a0) - phi(a1))) or (a0 - a1) == 0:
        a = a0
        return a
 
    d1 = dphi(a0) + dphi(a1) - 3 * (phi(a0) - phi(a1)) / (a0 - a1)
    if np.isnan(np.sign(a1 - a0) * np.sqrt(d1 ** 2 - dphi(a0) * dphi(a1))):
        a = a0
        return a
    d2 = np.sign(a1 - a0) * np.sqrt(d1 ** 2 - dphi(a0) * dphi(a1))
    a = a1 - (a1 - a0) * (dphi(a1) + d2 - d1) / (dphi(a1) - dphi(a0) + 2 * d2)

    return a


def wolfesearch(f, df, x0, p0, amax, c1, c2):
    a = amax
    aprev = 0
    phi = lambda x: f(x0 + x * p0)
    dphi = lambda x: np.dot(p0.transpose(), df(x0 + x * p0))

    phi0 = phi(0)
    dphi0 = dphi(0)
    i = 1
    imax = 1000
    while i < imax:
        if (phi(a) > phi0 + c1 * a * phi0) or ((phi(a) >= phi(aprev)) and (i > 1)):
            a = zoom(phi, dphi, aprev, a, c1, c2)
            return a

        if abs(dphi(a)) <= -c2 * dphi0:
            return a  # a is found already

        if dphi(a) >= 0:
            a = zoom(phi, dphi, a, aprev, c1, c2)
            return a

        a = cinterp(phi, dphi, a, amax)
        i += 1
    return a


def dfpsearch(f, df, x0, tol):
    c1 = tol
    c2 = 0.1
    amax = 3

    Hk = np.eye(len(x0))  
    k = 0  
    neval = 0 
    coords = [] 
    while True:
        asd = df(x0)
        pk = -np.dot(Hk, asd) 
        alpha = wolfesearch(f, df, x0, pk, amax, c1, c2)
        x1 = x0 + alpha * pk 

        dk = alpha * pk
        yk = df(x1) - df(x0)
        yk = yk.reshape((len(x0), 1))
# DFP
        #Hk = Hk + np.dot(dk, dk.transpose()) / np.dot(dk.transpose(), yk) - np.dot(np.dot(Hk, yk), np.dot(Hk, yk).transpose()) / np.dot(np.dot(yk.transpose(), Hk), yk)
# BFGS
        ykT_dk = np.dot(yk.transpose(), dk)
        gamma_k = 1.0 / ykT_dk
        term1 = np.outer(dk, dk) * gamma_k
        Hk_yk = np.dot(Hk, yk)
        term2 = np.outer(Hk_yk, Hk_yk) / np.dot(yk.transpose(), Hk_yk)
        Hk = Hk + term1 - term2



        coords.append(x0)

        if (norm(dk) < tol) or (k >= 1000):
            xmin = x1
            fmin = f(x1)
            return [xmin, fmin, neval, coords]

        x0 = x1  
        k += 1
        neval += 1 

X_test = np.array([1.0, 2.0])
print("fSphere(X_test) =", fSphere(X_test))  # Должно быть 1^2 + 2^2 = 5
print("dfSphere(X_test) =", dfSphere(X_test))  # Должно быть [2*1, 2*2] = [2, 4]

print(dfpsearch(fSphere,dfSphere,[1,0],10e-5))