import numpy as np
import sys
import math

def gradient (vec):
    n = len(vec)
    M = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1, n):
            diff = vec[j] - vec[i]
            M[i][j] = diff
            M[j][i] = -1 * diff
    return M

print(gradient([4,3,1]))
