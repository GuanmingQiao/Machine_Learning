# -*- coding: utf-8 -*-
"""
Spyder Editor

HW1 P5 KNN.
"""
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def loadDataset(filename, split):
    mat = scipy.io.loadmat(filename)
    data_X = mat['X']
    data_Y = mat['Y']
    data_X[data_X > 0] = 1
    msk = np.random.rand(10000) < split
    return (data_X[msk],data_X[~msk],data_Y[msk],data_Y[~msk])

def knnPrediction(trainingSet_X, trainingSet_Y, x, k, metric):
    dists = None
    if metric == '1':
        dists = np.sum(np.abs(trainingSet_X - x), axis=-1)
    elif metric == '2':
        dists = np.sum(np.abs(trainingSet_X - x)**2, axis=-1)**(1./2)
    else:
        dists = np.amax((trainingSet_X - x), axis=0)
    sorted_dists = sorted(dists)
    votes = []
    for i in range(k):
        index = np.where(dists == sorted_dists[i])[0][0]
        votes.append(trainingSet_Y[index][0])
    return findLabel(votes)

def findLabel(votes):
    b_count = 0
    b_label = -1
    for i in range(10):
        count = votes.count(i)
        if (count > b_count):
            b_count = count
            b_label = i
    return b_label

def test(trainingSet_X, trainingSet_Y, testingSet_X, testingSet_Y, k, nNorm):
    total = len(testingSet_X)
    correct = 0
    for i in range(total):
        prediction = knnPrediction(trainingSet_X, trainingSet_Y, testingSet_X[i], k, nNorm)
        actual = testingSet_Y[i]
        correct += (prediction == actual)
    return correct/total

def plot(trainingSet_X, testingSet_X, trainingSet_Y, testingSet_Y, ks, norms):
    errors = []
    for norm in norms:
        error = []
        for k in ks:
            err = 1 - test(trainingSet_X, trainingSet_Y, testingSet_X, testingSet_Y, k, norm)
            error.append(err)
            print(norm, k, err)
        errors.append(error)
    for i in range(len(norms)):
        plt.plot(ks, errors[i], label=norms[i])
    plt.title('K vs Error Rate')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.show()

def main():
    trainingSet_X=[]
    testingSet_X=[]
    trainingSet_Y=[]
    testingSet_Y=[]
    split = 0.8
    (trainingSet_X,testingSet_X, trainingSet_Y, testingSet_Y) = loadDataset('hw1data.mat', split)
    print ('Train set X: ' + repr(len(trainingSet_X)))
    print ('Train set Y: ' + repr(len(trainingSet_Y)))
    print ('Test set X: ' + repr(len(testingSet_X)))
    print ('Test set Y: ' + repr(len(testingSet_Y)))
    norms = ['1','2','infinity']
    ks = [4, 5, 6, 7, 8]
    plot(trainingSet_X, testingSet_X, trainingSet_Y, testingSet_Y, ks, norms)

main()