import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sys
import pdb

def norm1D(my,Sgm,x):

    [n,d]=np.shape(x)
    p = np.zeros(np.shape(x))
    for i in np.arange(0, n):
        p[i] = 1 / (np.sqrt(2 * np.pi) * Sgm) * \
            np.exp(-1 / 2 * np.square((x[i] - my)) / (np.square(Sgm)))

    return p

def norm2D(my,Sgm,x1,x2):
    X1,X2 = np.meshgrid(x1,x2)
    [n, d]=np.shape(X1)
    p = np.zeros(np.shape(X1))
    for i in np.arange(0, n):
        for j in np.arange(0,d):
            xij = np.array([[X1[i,j]],[X2[i,j]]])
            dist = xij - my
            distT = np.transpose(dist) 
            denominator = 2 * np.pi * np.sqrt(np.linalg.det(Sgm))
            exponent = - 1/2 * distT @ np.linalg.inv(Sgm) @ dist
            p[i][j] = 1 / denominator * np.exp(exponent)

    return p,X1,X2
    

def knn2D(X, k, x1, x2):
    X1,X2 = np.meshgrid(x1,x2)
    [n, d]=np.shape(X1)
    p = np.zeros(np.shape(X1))
    for i in range(n):
        for j in range(d):
            D = []
            for ind in range(X.shape[1]):
                D.append(np.linalg.norm(np.array([X[0][ind] - X1[i][j], X[1][ind] - X2[i,j] ])))
            D.sort()
            dk = D[k-1]
            p[i,j]= k/(np.pi*pow(dk,2)*X.shape[1])
    return p,X1,X2