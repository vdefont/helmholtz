import numpy as np
import sys
import math
import csv
import util

def multiplyMatrices(matrices):
    matrices.reverse()
    M = np.matmul(matrices[1], matrices[0])
    for i in range(2, len(matrices)):
        M = np.matmul(matrices[i], M)
    return M

A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
C = np.array([[9,10],[11,12]])
d = np.array([[13],[14]])
print(np.matmul(A, np.matmul(B, np.matmul(C, d))))
print(multiplyMatrices([A,B,C,d]))
