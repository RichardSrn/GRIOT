from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class SEP():
    """
    Artificially SEParate the communities in a 2D space.
    The dimensionality reduction is done with LDA (mainly for speed).
    Then the i-th community is centered in (0,0), multiplied by r2 and projected on
    the i-th point of a circle of radius r1 and angle i/n*2jπ.
    """
    def __init__(self, n_components=2, r1=4, r2=1, keep_proportions=False):
        self.n_components = n_components
        self.r1 = r1
        self.r2 = r2
        self.keep_proportions = keep_proportions
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)

    def fit_transform(self, X, y):
        F = self.lda.fit_transform(X, y)

        # set n the number of classes
        n = len(np.unique(y))

        if self.r2 <= 0:
            # get the maximum variance between all the classes
            var = np.array([np.var(F[y==i , :], axis=0).max() for i in range(n)])
            var = [max(var)**.5 for _ in range(n)]
        elif self.keep_proportions:
            # get the maximum variance between all the classes
            var = np.array([np.var(F[y==i , :], axis=0).max() for i in range(n)])
            var /= var.max()
            var *= self.r2
        else:
            var = [self.r2 for _ in range(n)]

        for i in range(n):
            coordinates = self.r1 * np.exp(i * 2 * np.pi * 1j / n)
            coord_x = np.real(coordinates)
            coord_y = np.imag(coordinates)
            # center the current class in 0,0
            F[y == i, 0] -= np.mean(F[y == i, 0])
            F[y == i, 1] -= np.mean(F[y == i, 1])
            # # fit the current class in a circle of radius 1 in both axis
            # F[y == i, 0] /= np.max(np.abs(F[y == i, 0]))
            # F[y == i, 1] /= np.max(np.abs(F[y == i, 1]))
            # multiply the coordinates of the current class by r2 and the variance to keep proportions
            # note, if keep_proportions is False, var[i] = 1
            F[y == i, 0] *= var[i]
            F[y == i, 1] *= var[i]
            # add coord_x to the first column of F and coord_y to the second column of F
            F[y == i, 0] += coord_x
            F[y == i, 1] += coord_y
        return F
