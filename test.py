import numpy as np
import sys
import math
import csv
import util

# Remove entries where weight is 0
# dim: 0 is rows, 1 is cols
def removeZeroEdges(x, wFull, dim=0):

    x2 = []

    for i in range(len(wFull)):
        if wFull[i] != 0:

            # 1D array or removing rows
            if len(x.shape) == 1 or dim == 0:
                x2.append(x[i])
            # Removing cols
            else:
                x2.append(x[:,i])

    x2 = np.array(x2)
    if dim == 1:
        x2 = x2.transpose()
    return x2

# Assumes vec is only the upper triangle
# 'Symmetric' is either 1 (sym) or -1 (skew-sym)
def vecToMatrix (v, symmetric=1):
    n = int((1 + math.sqrt(1 + 8 * len(v))) / 2)
    M = np.zeros((n,n))
    c = 0
    for i in range(n-1):
        for j in range(i+1, n):
            M[i][j] = v[c]
            M[j][i] = symmetric * v[c]
            c += 1
    return M

wFull = np.array([0,1,1])
x1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(removeZeroEdges(x1,wFull,1))
print(removeZeroEdges(x1,wFull,0))
