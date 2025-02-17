# Gauss-Seidel Method

import numpy as np

def gauss_seidel(a:np.ndarray, b:np.ndarray, x:np.ndarray, iter = 0):
    for _ in range(iter):
        x_new = np.empty(x.shape)
        for i in range(x.shape[0]):
            t = x.copy()
            t[i] = 0
            ax = a[i]@t
            x_new[i] = (b[i] - ax) / a[i,i]
        x = x_new
        print(x)
    
    return x
    
if __name__ == "__main__":
    
    a = np.array([[ 5,  1,  1,],
                  [-2,  4,  0,],
                  [ 0,  2,  4,],])
    
    b = np.array([6, -22,  -4])
    x_init = np.zeros(b.shape)
    
    x = gauss_seidel(a, b, x_init, 10)