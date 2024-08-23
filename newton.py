import numpy as np


def f(x): return x**2 - 10*np.cos(0.3*np.pi*x) - 20
def df(x): return 2*x + 3*np.pi*np.sin(0.3*np.pi*x)
def ddf(x): return 2 + 0.9*(np.pi**2)*np.cos(0.3*np.pi*x)

tol = 0.01



def nsearch(tol, x0):

    x = x0
    neval = 0
    coords = [x]
    
    neval += 3
    x_new = x - df(x) / ddf(x)
    coords.append(x_new)
    

    while np.abs(df(x_new)) > tol:
        x = x_new
        neval += 3
        x_new = x - df(x) / ddf(x)
        coords.append(x_new)
    

    xmin = x_new
    fmin = f(xmin)
    
    return [xmin, fmin, neval, coords]
