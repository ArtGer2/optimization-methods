import numpy as np
def f(x): return 2 * (x ** 2) - 9 * x - 31
def df(x): return 4 * x - 9

def bsearch(interval, tol):
    neval = 0
    [a, b] = interval
    while (np.abs(b-a) > tol and (np.abs(df(a)) > tol)): 
        mid = (a + b)/2

        if (df(mid) > 0): 
            b = mid
        else:
            a = mid

        interval.append(mid)
        neval += 2

    return [mid, f(mid), neval + 1, interval]