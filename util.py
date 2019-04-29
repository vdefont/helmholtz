import numpy as np
import random
import matplotlib.pyplot as plt

# Multiplies a chain of matrices, right to left
def multiplyMatrices(matrices):
    matrices.reverse()
    M = np.matmul(matrices[1], matrices[0])
    for i in range(2, len(matrices)):
        M = np.matmul(matrices[i], M)
    return M

# v1, v2 are vectors
# w are weights
def innerProd(v1, v2, w=None):
    if w is None:
        return sum(v1 * v2)
    return sum(v1 * v2 * w)

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

# Draws with replacement - mutates
def randomizeVector(v):
    n = len(v)
    tmp = []
    for i in range(n):
        r = random.randint(0, n-1)
        tmp.append(v[r])
    for i in range(n):
        v[i] = tmp[i]

def plotHistogram():
    x = np.genfromtxt('output/varianceData/tennis50GradCurlHarm.csv', delimiter=',')

    plt.hist(x[:,0], bins=30)
    plt.axvline(0.35177379571732653, color='k', linestyle='dashed', linewidth=1)
    plt.show()

    plt.hist(x[:,1], bins=30)
    plt.axvline(0.645027535536416, color='k', linestyle='dashed', linewidth=1)
    plt.show()

    plt.hist(x[:,2], bins=30)
    plt.axvline(0.00315854160653562, color='k', linestyle='dashed', linewidth=1)
    plt.show()

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
