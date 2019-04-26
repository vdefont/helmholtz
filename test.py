import numpy as np
import sys
import math
import csv
import util

# Returns dict with top players as keys, rank as index (starts at 0)
def getTopTennisPlayers(numPlayers):
    file = open('data/tennis/ranking')
    reader = csv.reader(file, delimiter = ',')
    topPlayers = {}
    curI = 0
    for lineItems in reader:
        name = lineItems[0]
        nameFormatted = " ".join(name.split())
        topPlayers[nameFormatted] = curI
        curI += 1
        if len(topPlayers) >= numPlayers:
            break
    return topPlayers

for i in range(10):
    v = [1,1,1,1,2,3,4,5]
    util.randomizeVector(v)
    print(v)
