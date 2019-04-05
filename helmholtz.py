import numpy as np
import sys
import os

import fileReader

# TODO:
# - Compare to rankings online (chrome bookmark)
# - Try different winMargin methods in fileReader (ex: binary)
# - weight more for most recent dates?
# - Try different averaging mechanisms

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

    users = fileReader.loadTennisData()
    Y, W, itemIndices = makeMatrices(users)
    s = solve(Y, W)

    # Associate items to values
    itemValues = {}
    for item in itemIndices:
        itemValues[item] = s[itemIndices[item]]

    writeFile(itemValues, outputFile)
