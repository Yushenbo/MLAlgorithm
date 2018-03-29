#-*- coding:utf-8 -*-
#########################################################################
# File Name: sklearn-knn-demo.py
# Author: Nichol.Shen
# mail: nichol_shen@yahoo.com
#########################################################################
#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from numpy import *
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 3

# import data that we perform
# iris = datasets.load_iris()
# X = iris.data[:, :2] # We only used the former 2 fetures 
# y = iris.target

# print('X = ', type(X), X)
# print('Y = ', type(Y), Y)

X = array([[-1.0, -1.1], [-1.0, -1.0], [0, 0], [1.0, 1.1],
    [2.0, 2.0], [2.0, 2.1]])
y = array([0, 0 , 0, 1, 1, 1])

h = .02

# Create calorful map
cmap_light = ListedColormap(['#ffaaaa', '#aaffaa'])
cmap_light = ListedColormap(['#ff0000', '#00ff00'])

for weights in ['uniform', 'distance']:
# Create  a KNN clsssifier instance and suit the data
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Draw the ehge of decision allocation colors
    # Draw center points

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h))


    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # plan result in a colored map
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Draw tainning point
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('3-Class classificaiton (k = %i, weights = "%s")'
            %(n_neighbors, weights))


plt.show()


