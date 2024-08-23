import numpy as np

def f(x): return x**2 -  10*np.cos(0.3*np.pi*x) - 20
def df(x): return 2*x + 3*np.pi*np.sin(0.3*np.pi*x)

def ssearch(interval, tol):
    x0, x1 = interval
    dfx0 = df(x0)
    dfx1 = df(x1)
    x2 = x1 - dfx1 *(x1-x0)/ (dfx1 - dfx0)
    dfx2 = df(x2)

    neval = 3
    coords = [[x2, x0, x1]]

    if dfx2 > 0:
        x1 = x2
    else:
        x0 = x2

    while (np.abs(dfx1) > tol) and (np.abs(x1 - x0) > tol):

        x_next = x1 - dfx1 * (x1 - x0) / (dfx1 - dfx0)
        dfN = df(x_next)
            

        if dfN > 0:
            x1 = x_next
            dfx1 = dfN
        else:
            x0 = x_next
            dfx0 = dfN
        
        neval +=1
        coords.append([x_next, x0, x1])

    return [x_next, f(x_next), neval, coords]

print(ssearch([-2,5],0.1))
