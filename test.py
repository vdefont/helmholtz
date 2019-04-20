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

def makeGradientMatrix (n):
    M = np.zeros((n*n,n))
    c1 = 0
    for i in range(n):
        c2 = 0
        for j in range(n):
            row = n*i + j
            M[row][c1] -= 1
            M[row][c2] += 1
            c2 += 1
        c1 += 1
    return M

# Input: a matrix of weights
def makeDivMatrix (W):
    n = len(W)
    M = np.zeros((n,n*n))
    for row in range(n):
        for col in range(n):
            realCol = row + n * col
            M[row][realCol] = W[row][col]
    return M

print(makeGradientMatrix(4))

W = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
print(makeDivMatrix(W))
