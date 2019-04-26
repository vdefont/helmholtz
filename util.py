import numpy as np
import random
import matplotlib.pyplot as plt


# Draws with replacement
def randomizeVector(v):
    n = len(v)
    tmp = []
    for i in range(n):
        r = random.randint(0, n-1)
        tmp.append(v[r])
    for i in range(n):
        v[i] = tmp[i]

def plotHistogram():
    x = np.genfromtxt('output/varianceData/tennis50HarmOverCurl.csv', delimiter=',')
    plt.hist(x, bins=30)
    plt.show()

# plotHistogram()
