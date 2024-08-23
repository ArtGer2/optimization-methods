import numpy as np

def f(x): return (x - 3)**2- 3*x + x**2 - 40

def gsearch(interval, tol):

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

    answer_ = [xmin, f(xmin), neval, coords]
    return answer_