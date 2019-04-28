import numpy as np
import sys
import os
import math
import scipy.sparse.linalg

import fileReader
import util

# Returns:
# - Comparison matrix (skew-symmetric)
# - Weight matrix (symmetric)
# - Item indices
def makeMatrices(users, weights = None, itemIndices = None):

    # Compile a list of all items that were compared
    if itemIndices is None:
        itemIndices = {}
        curI = 0
        for user in users:
            if len(user) > 1: # If 1 or less, then no comparisons were made
                for item in user:
                    if item not in itemIndices:
                        itemIndices[item] = curI
                        curI += 1

    numItems = len(itemIndices)
    dims = (numItems, numItems)
    Y = np.zeros(dims) # Comparison matrix
    W = np.zeros(dims) # Weight matrix
    Counts = np.zeros(dims) # Count number of entries at each index (i,j)

    # Add user data to comparison and weight matrix
    for userI in range(len(users)):
        user = users[userI]
        userItems = list(user.keys())

        # Loop through all pairs
        for i in range(len(userItems)):
            for j in range(i+1, len(userItems)):
                item1 = userItems[i]
                item2 = userItems[j]
                item1I = itemIndices[item1]
                item2I = itemIndices[item2]

                # Update weights matrix
                weight = weights[userI] if weights else 1
                W[item1I][item2I] += weight
                W[item2I][item1I] += weight

                # Update ratings matrix
                item1Score = user[item1]
                item2Score = user[item2]
                Y[item1I][item2I] += item2Score - item1Score
                Y[item2I][item1I] += item1Score - item2Score

                # Update counts matrix
                Counts[item1I][item2I] += 1
                Counts[item2I][item1I] += 1

    # Average Y - divide by counts
    for i in range(numItems):
        for j in range(numItems):
            curCount = Counts[i][j]
            if curCount != 0:
                Y[i][j] /= curCount

    return (Y, W, itemIndices)

# Only keeps upper triangle of matrix
def matrixToVec (M):
    n = len(M)
    vec = []
    for i in range(n-1):
        for j in range(i+1, n):
            vec.append(M[i][j])
    return np.array(vec)

# Returns solution in (im gradient) that minimizes error
def solve(Y, W):

    # Make laplacian
    laplacian = -1 * W
    rowSums = np.sum(W, axis=1)
    for i in range(len(rowSums)):
        laplacian[i][i] = rowSums[i]

    # Pseudo-invert laplacian
    pInvLap = np.linalg.pinv(laplacian)

    # Make divergence
    div = np.sum(Y * W, axis=1)

    # Solve
    s = np.sum(pInvLap * -1 * div, axis=1)

    return s

def getGradientFlow (vec, wFull):

    gradientFlow = []

    n = len(vec)
    ij = 0
    for i in range(n-1):
        for j in range(i+1, n):
            if wFull[ij] != 0:
                diff = vec[j] - vec[i]
                gradientFlow.append(diff)
            ij += 1

    return np.array(gradientFlow)

def doubleCumSum (n):
    sum = 0
    for e in range(n, 0, -1):
        sum += e * (n + 1 - e)
    return sum

def makeCurlMatrix (numVars):

    freeVars = numVars - 2

    nRow = doubleCumSum(freeVars)
    nCol = int(numVars * (numVars - 1) / 2)
    M = np.zeros((nRow, nCol))

    row = 0
    c1 = -1
    c2 = freeVars
    c3start = freeVars

    for top in range(freeVars, 0, -1):

        c3 = c3start
        c3start += top

        for cur in range(top, 0, -1):

            c1 += 1
            c2 -= cur

            for i in range(cur):
                c2 += 1
                c3 += 1

                M[row][c1] = 1
                M[row][c2] = -1
                M[row][c3] = 1

                row += 1

        c1 += 1
        c2 += top

    return M
def makeCurlAdjoint (curlM, w):
    return (curlM / w).transpose()

# Returns: map from singles to pairs
def makeGradientMatrix (n):
    rows = int(n * (n-1) / 2)
    M = np.zeros((rows,n))
    row = 0
    c1 = 0
    for blockSize in range(n-1,0,-1):
        c2 = c1 + 1
        for i in range(blockSize):
            M[row][c1] = -1
            M[row][c2] = 1
            row += 1
            c2 += 1
        c1 += 1
    return M
def makeGradientAdjoint (gradM, w):
    return gradM.transpose() * w

def getCurlResidual(y, w, n, wFull):

    curlM = removeZeroEdges(makeCurlMatrix(n), wFull, dim=1)
    curlAdj = makeCurlAdjoint(curlM, w)

    # Multiply Y and curl by sqrt of weights before solving (to weight appropriately)
    yLsqr = y * (w ** 0.5)
    curlLsqr = curlAdj * np.array([w ** 0.5]).transpose()

    # Solve
    lsqrTriples = scipy.sparse.linalg.lsqr(curlLsqr, yLsqr)[0]
    curlResidual = np.matmul(curlAdj, lsqrTriples)
    return curlResidual

def getHelmholtzian(y, w, n, wFull):
    curl = removeZeroEdges(makeCurlMatrix(n), wFull, dim=1)
    curlAdj = makeCurlAdjoint(curl, w)
    grad = removeZeroEdges(makeGradientMatrix(n), wFull, dim=0)
    gradAdj = makeGradientAdjoint(grad, w)
    helmholtzian = np.matmul(curlAdj, curl) + np.matmul(grad, gradAdj)
    return helmholtzian

def getHarmonicResidual(y, w, n, wFull):
    S = getHelmholtzian(y, w, n, wFull)

    Dpos = np.diag(w ** 0.5)
    Dneg = np.diag(w ** -0.5)

    S2 = np.matmul(Dpos, np.matmul(S, Dneg))
    S2inv = np.linalg.pinv(S2)

    proj = np.matmul(np.matmul(Dneg,S2inv), np.matmul(S2, Dpos))

    harmonicResidual = np.matmul(proj, y)
    return harmonicResidual

def writeFile(itemValues, fileName):
    sortedItems = sorted(itemValues, key=itemValues.get, reverse=True)
    with open(fileName, 'w') as file:
        for sortedItem in sortedItems:
            line = ",".join([sortedItem, str(itemValues[sortedItem])])
            file.write(line)
            file.write('\n')

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

# Returns Y, W, Grad, Curl, Harm flows
# - randomize=True - performs bootstrapping to randomly sample data
# - verbose=True - prints out details about 3 flows
def getFlows(data, randomize=False, verbose=False):
    # Unpack args
    (users, weights, order) = data

    Y, W, itemIndices = makeMatrices(users, weights, order)
    n = len(Y)
    yFull = matrixToVec(Y)
    wFull = matrixToVec(W)

    if randomize:
        util.randomizeVector(yFull)
        util.randomizeVector(wFull)

    y = removeZeroEdges(yFull, wFull)
    w = removeZeroEdges(wFull, wFull)

    # Get three flows
    s = solve(Y, W)
    gradientFlow = getGradientFlow(s, wFull)
    curlResidual = getCurlResidual(y, w, n, wFull)
    harmonicResidual = getHarmonicResidual(y, w, n, wFull)

    # TODO remove - temporary override harm flow calc
    # harmonicResidual = Y - gradientFlow - curlResidual

    # Associate items to values
    itemValues = {}
    for item in itemIndices:
        itemValues[item] = s[itemIndices[item]]

    writeFile(itemValues, outputFile)

    if verbose:
        print("\nG")
        print(gradientFlow)
        print("\nC")
        print(curlResidual)
        print("\nH")
        print(harmonicResidual)
        print("\nW")
        print(w)

        print("\nDot products in W-space -- should be 0 (orthogonal flows)")
        print("g c -- " + str(sum(gradientFlow * curlResidual * w)))
        print("g h -- " + str(sum(gradientFlow * harmonicResidual * w)))
        print("c h -- " + str(sum(curlResidual * harmonicResidual * w)))

    return y, w, gradientFlow, curlResidual, harmonicResidual

def getSquaredNorm(v, innerProd=None):
    if innerProd is None:
        return sum(v * v)
    return sum(v * v * innerProd)

pyVersion = sys.version_info[0]
if pyVersion < 3:
    print("Please use at least python3")
elif len(sys.argv) < 2:
    print("Please provide the following args:")
    print("- outputFile")
else:
    outputFileName = sys.argv[1]
    outputFile = "output" + os.sep + outputFileName

    # Tennis
    # weightMethod = "GAMES"
    # maxPlayers = 50
    # data = fileReader.loadTennisData(weightMethod = weightMethod, maxPlayers=maxPlayers)

    # Golf
    # maxPlayers = 10
    # data = fileReader.loadGolfData(maxPlayers)

    # Test
    data = fileReader.testData2()

    # print("G\t\t\tC\t\t\tH")

    randomize=False
    for i in range(1):
        y, w, g, c, h = getFlows(data, randomize=randomize, verbose=True)
        randomize=True # Randomize everything after first one

        yNorm = getSquaredNorm(y, w)
        gNorm = getSquaredNorm(g, w)
        cNorm = getSquaredNorm(c, w)
        hNorm = getSquaredNorm(h, w)

        print(str(gNorm/yNorm) + "\t" + str(cNorm/yNorm) + "\t" + str(hNorm/yNorm))
