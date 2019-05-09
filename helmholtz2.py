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

# For curl matrix, only the triples which don't contain zero edges
def getValidTripleIndices(curlFull, wFull):
    rows, cols = curlFull.shape
    validTriples = np.ones(rows)
    for r in range(rows):
        for c in range(cols):
            if curlFull[r][c] != 0 and wFull[c] == 0:
                validTriples[r] = 0
                break
    return validTriples
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

def getGradFlow(matrices):
    y, w, grad, gradAdj, curl, curlAdj = matrices

    lapInv = np.linalg.pinv(np.matmul(gradAdj, grad)) # Inv of laplacian
    s = util.multiplyMatrices([lapInv, gradAdj, y])
    gradFlow = np.matmul(grad, s)
    return gradFlow, s

def getCurlFlow(matrices):
    y, w, grad, gradAdj, curl, curlAdj = matrices

    yLsqr = y * (w ** 0.5)
    cLsqr = curlAdj * np.array([w ** 0.5]).transpose()

    lsqrTriples = scipy.sparse.linalg.lsqr(cLsqr, yLsqr)[0]
    curlFlow = np.matmul(curlAdj, lsqrTriples)
    return curlFlow

def getHarmFlow(matrices):
    y, w, grad, gradAdj, curl, curlAdj = matrices

    S = np.matmul(curlAdj, curl) - np.matmul(grad, gradAdj)
    Dpos = np.diag(w ** 0.5)
    Dneg = np.diag(w ** -0.5)
    S2 = util.multiplyMatrices([Dpos, S, Dneg])
    S2pinv = np.linalg.pinv(S2)
    harmFlow = y - util.multiplyMatrices([Dneg, S2pinv, S2, Dpos, y])

    return harmFlow

def writeItemScoresToFile(s, itemIndices, fileName):
    itemValues = {}
    for item in itemIndices:
        itemValues[item] = s[itemIndices[item]]

    sortedItems = sorted(itemValues, key=itemValues.get, reverse=True)
    with open(fileName, 'w') as file:
        for sortedItem in sortedItems:
            line = ",".join([sortedItem, str(itemValues[sortedItem])])
            file.write(line)
            file.write('\n')

# Returns Y, W, Grad, Curl, Harm flows
# - randomizeMethod =
#       <bootstrap> - performs bootstrapping to randomly sample data
#       <normal>    - draws from normal dist with same mean and stdev as data
# - gradOnly=True - only compute grad flow. Set others to 0. (Much faster)
def getFlows(data, randomizeMethod=None, gradFlowOnly=False, gradFlowFile=None):
    (users, weights, order) = data # Unpack args
    Y, W, itemIndices = makeMatrices(users, weights, order)
    n = len(Y)
    yFull = matrixToVec(Y)
    wFull = matrixToVec(W)

    if randomizeMethod:
        util.randomizeVector(yFull, randomizeMethod)
        # Only use positive values for weight
        util.randomizeVector(wFull, randomizeMethod, positiveVals=True)

    y = util.removeZeroEdges(yFull, wFull)
    w = util.removeZeroEdges(wFull, wFull)

    gradFull = makeGradientMatrix(n)
    grad = util.removeZeroEdges(gradFull, wFull, dim=0)
    gradAdj = makeGradientAdjoint(grad, w)

    if gradFlowOnly:
        curl = 0
        curlAdj = 0
    else:
        curlFull = makeCurlMatrix(n)
        validTriples = getValidTripleIndices(curlFull, wFull)
        curl = util.removeZeroEdges(curlFull, wFull, dim=1) # Keep valid edges
        curl = util.removeZeroEdges(curl, validTriples, dim=0) # Keep valid triples
        curlAdj = makeCurlAdjoint(curl, w)

    matrices = (y, w, grad, gradAdj, curl, curlAdj)
    gradFlow, s = getGradFlow(matrices)
    if gradFlowOnly:
        curlFlow = np.zeros(gradFlow.shape)
        harmFlow = np.zeros(gradFlow.shape)
    else:
        curlFlow = getCurlFlow(matrices)
        harmFlow = getHarmFlow(matrices)

    if gradFlowFile is not None:
        writeItemScoresToFile(s, itemIndices, gradFlowFile)

    return y, w, gradFlow, curlFlow, harmFlow

def printFlowFractions(flows):
    y, w, gradFlow, curlFlow, harmFlow = flows
    yNormSq = util.innerProd(y, y, w)
    gNormSq = "%4f" % (util.innerProd(gradFlow, gradFlow, w) / yNormSq)
    cNormSq = "%4f" % (util.innerProd(curlFlow, curlFlow, w) / yNormSq)
    hNormSq = "%4f" % (util.innerProd(harmFlow, harmFlow, w) / yNormSq)
    print(gNormSq + "\t" + cNormSq + "\t" + hNormSq)

pyVersion = sys.version_info[0]
if pyVersion < 3:
    print("Please use at least python3")
else:
    # Tennis
    # maxPlayers = 50
    # weightMethod = "GAMES"
    # data = fileReader.loadTennisData(weightMethod = weightMethod, maxPlayers=maxPlayers)

    # Test
    # data = fileReader.testData4()

    # Chess
    # maxPlayers = 50
    # data = fileReader.loadChessData(50)

    # flows = getFlows(data, randomize=randomize)
    # printFlowFractions(flows)

    print("\nRANDOMIZING 50-PLAYER GOLF DATA WITH BOOTSTRAP")
    maxPlayers = 50
    randomizeMethod=None
    for i in range(1):
        data = fileReader.loadGolfData(maxPlayers)
        flows = getFlows(data, randomizeMethod=randomizeMethod)
        randomizeMethod="bootstrap"
        printFlowFractions(flows)

    print("\nRANDOMIZING 50-PLAYER GOLF DATA WITH NORMAL")
    maxPlayers = 50
    randomizeMethod=None
    for i in range(1):
        data = fileReader.loadGolfData(maxPlayers)
        flows = getFlows(data, randomizeMethod=randomizeMethod)
        randomizeMethod="normal"
        printFlowFractions(flows)

    print("\nRANDOMIZING 50-PLAYER TENNIS DATA WITH NORMAL")
    maxPlayers = 50
    randomizeMethod=None
    for i in range(1):
        data = fileReader.loadTennisData("GAMES", maxPlayers)
        flows = getFlows(data, randomizeMethod=randomizeMethod)
        randomizeMethod="normal"
        printFlowFractions(flows)
