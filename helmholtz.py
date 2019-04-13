import numpy as np
import sys
import os
import math
import scipy.sparse.linalg

import fileReader

# TODO:
# - Figure out residual flows - esp. curl
# - Compare
#   - to rankings online (chrome bookmark)
#   - to PCA
# - diff. methods
#   - Try different winMargin methods in fileReader (ex: binary)
#   - weight more for most recent dates?
#   - Try different averaging mechanisms

# Returns:
# - Comparison matrix (skew-symmetric)
# - Weight matrix (symmetric)
# - Item indices
def makeMatrices(users):

    # Compile a list of all items that were compared
    itemIndices = {}
    curI = 0
    for user in users:
        if len(user) > 1: # If 1 or less, then no comparisons were made
            for item in user:
                if item not in itemIndices:
                    itemIndices[item] = curI
                    curI += 1

    numItems = len(itemIndices)
    Y = np.array([[0.0] * numItems] * numItems) # Comparison matrix
    W = np.array([[0.0] * numItems] * numItems) # Weight matrix

    # Add user data to comparison and weight matrix
    for user in users:
        userItems = list(user.keys())

        # Loop through all pairs
        for i in range(len(userItems)):
            for j in range(i+1, len(userItems)):
                item1 = userItems[i]
                item2 = userItems[j]
                item1I = itemIndices[item1]
                item2I = itemIndices[item2]

                # Update weights matrix
                W[item1I][item2I] += 1
                W[item2I][item1I] += 1

                # Update ratings matrix
                item1Score = user[item1]
                item2Score = user[item2]
                Y[item1I][item2I] += item2Score - item1Score
                Y[item2I][item1I] += item1Score - item2Score


    Y = np.array(Y)
    W = np.array(W)

    # Divide Y by weights
    for i in range(numItems):
        for j in range(numItems):
            curW = W[i][j]
            if curW != 0:
                Y[i][j] /= curW

    return (Y, W, itemIndices)

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

def gradient (vec):
    n = len(vec)
    M = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1, n):
            diff = vec[j] - vec[i]
            M[i][j] = diff
            M[j][i] = -1 * diff
    return M

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

# Useful for handling curl
def matrixToVec (M):
    n = len(M)
    vec = []
    for i in range(n-1):
        for j in range(i+1, n):
            vec.append(M[i][j])
    return np.array(vec)
def vecToMatrix (v):
    n = int((1 + math.sqrt(1 + 8 * len(v))) / 2)
    M = np.zeros((n,n))
    c = 0
    for i in range(n-1):
        for j in range(i+1, n):
            M[i][j] = v[c]
            M[j][i] = -1 * v[c]
            c += 1
    return M

def getCurlResidualOld(Y, W):
    curlM = makeCurlMatrix(len(Y))
    curlMpInv = np.linalg.pinv(curlM)
    yVec = matrixToVec(Y)
    curlResidualVec = np.matmul(curlMpInv, np.matmul(curlM, yVec))
    return vecToMatrix(curlResidualVec)

def getCurlResidual(Y, W):

    curlM = makeCurlMatrix(len(Y))

    weightVec = matrixToVec(W)
    # Replace 0 by 1 in sqrt matrix to avoid dividing by 0
    for i in range(len(weightVec)):
        if weightVec[i] == 0:
            weightVec[i] = 1

    curlAdj = curlM.transpose() / np.array([weightVec]).transpose()

    # Multiply Y and curl by sqrt of weights before solving (to weight appropriately)
    Ylsqr = matrixToVec(Y) * (weightVec ** 0.5)
    curlLsqr = curlAdj * np.array([weightVec ** 0.5]).transpose()

    # Solve
    lsqrTriples = scipy.sparse.linalg.lsqr(curlLsqr, Ylsqr)[0]

    curlResidualVec = np.matmul(curlAdj, lsqrTriples)
    return vecToMatrix(curlResidualVec)

def writeFile(itemValues, fileName):
    sortedItems = sorted(itemValues, key=itemValues.get, reverse=True)
    with open(fileName, 'w') as file:
        for sortedItem in sortedItems:
            line = ",".join([sortedItem, str(itemValues[sortedItem])])
            file.write(line)
            file.write('\n')

pyVersion = sys.version_info[0]
if pyVersion < 3:
    print("Please use at least python3")
elif len(sys.argv) < 1:
    print("Please provide the following args:")
    print("- outputFile")
else:
    outputFileName = sys.argv[1]
    outputFile = "output" + os.sep + outputFileName

    maxRows = 90
    users, order = fileReader.loadTennisData(maxRows)

    Y, W, itemIndices = makeMatrices(users)
    s = solve(Y, W)
    gradientFlow = gradient(s)

    # Associate items to values
    itemValues = {}
    for item in itemIndices:
        itemValues[item] = s[itemIndices[item]]

    writeFile(itemValues, outputFile)

    # Solve least squares: Y = curl* X
    curlResidual = getCurlResidual(Y, W)

    harmonicResidual = Y - gradientFlow - curlResidual

    print("\nNorms:")
    print("Gradient: " + str(np.linalg.norm(gradientFlow)))
    print("Curl:     " + str(np.linalg.norm(curlResidual)))
    print("Harmonic: " + str(np.linalg.norm(harmonicResidual)))
