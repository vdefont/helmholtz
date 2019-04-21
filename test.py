import numpy as np
import sys
import math
import csv

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

print(getTopTennisPlayers(5))
