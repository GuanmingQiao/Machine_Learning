#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 21:20:11 2018

@author: guanmingqiao
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA

def cluster_points(X, mu):
    clusters  = {}
    labels = []
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
            labels.append(bestmukey)
        except KeyError:
            clusters[bestmukey] = [x]
            labels.append(bestmukey)
    return clusters, labels

def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
    if (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])):
        return True
    return False

def find_centers(X, K):
    oldmu = random.sample(list(X), K)
    mu = random.sample(list(X), K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters, labels = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, labels)

def main():
    #preprocessing
    X, y = make_moons(200, noise=.05, random_state=0)
    scikit_kpca_rbf = KernelPCA(n_components=1, kernel='rbf', gamma=15)
    scikit_kpca_poly = KernelPCA(n_components=1, kernel='poly', gamma=-4, degree = 2)
    scikit_kpca_linear = KernelPCA(n_components=1, kernel='linear', gamma=-4)

    #run kmeans on different settings
    K_lloyd = X
    K_rbf = scikit_kpca_rbf.fit_transform(X)
    K_poly = scikit_kpca_poly.fit_transform(X)
    K_linear = scikit_kpca_linear.fit_transform(X)

    centers_lloyd, labels_lloyd = find_centers(K_lloyd, 2)
    centers_rbf, labels_rbf = find_centers(K_rbf, 2)
    centers_poly, labels_poly = find_centers(K_poly, 2)
    centers_linear, labels_linear = find_centers(K_linear, 2)


    #plotting
    plt.scatter(X[:, 0], X[:, 1],
            s=50, cmap='viridis')
    plt.title('original dataset')
    plt.show()

    plt.scatter(X[:, 0], X[:, 1], c=labels_lloyd,
            s=50, cmap='viridis')
    plt.title('unkernelized')
    plt.show()

    plt.scatter(X[:, 0], X[:, 1], c=labels_rbf,
            s=50, cmap='viridis')
    plt.title('rbf kernel')
    plt.show()

    plt.scatter(X[:, 0], X[:, 1], c=labels_poly,
            s=50, cmap='viridis')
    plt.title('poly kernel')
    plt.show()

    plt.scatter(X[:, 0], X[:, 1], c=labels_linear,
            s=50, cmap='viridis')
    plt.title('linear kernel')
    plt.show()


main()