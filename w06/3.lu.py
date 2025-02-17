# LU Decomposition
import numpy as np

def lu_decomposition_square(A:np.ndarray):
    demention = A.shape[0]
    L = np.eye(demention)
    U = A.copy()
    
    for i in range(demention):
        for j in range(i+1, demention):
            
            print(i, j)
            
            L[j, i] = U[j, i] / U[i, i]
            U[j] = U[j] - L[j,i] * U[i]
        
    return L, U
    
if __name__ == '__main__':
    a = np.array([[ 2,  4,  3,  5],
                  [-4, -7, -5, -8],
                  [ 6,  8,  2,  9],
                  [ 4,  9, -2, 14],])
    
    l, u = lu_decomposition_square(a)
    
    print(l)
    print(u)
    print(l@u)
    
