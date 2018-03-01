#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:15:51 2018

@author: guanmingqiao
"""
# Import Package
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from math import floor

def train(training_data, testing_data, passes, eta, Whidden, Bhidden, Wout, Bout):
    for p in range(passes):
        for x,y in training_data:
            #Forward Propogation
            hidden_arg = np.dot(Whidden, x) + Bhidden
            hidden_act = sig(hidden_arg)
            output_arg = np.dot(Wout, hidden_act) + Bout
            output_act = sig(output_arg)

            #Backpropagation
            delta = (output_act - y) * sigmoid(output_arg)
            delta2 = np.dot(Wout.transpose(), delta) * sigmoid(hidden_arg)
            bias = delta;
            bias2 = delta2;
            weights = delta.dot(hidden_act.transpose())
            weights2 = delta2.dot(x)
            Whidden -= eta * weights2
            Bhidden -= eta * bias2
            Wout -=  eta * weights
            Bout -= eta * bias

        current_error = estimate_error([Whidden, Wout],[Bhidden, Bout], testing_data)
        print(p, current_error, eta)
    return [[Whidden, Wout],[Bhidden, Bout]]

def estimate_error(W, B, testing_data):
    total_err = 0
    for x, y in testing_data:
        y_pred = predict(W, B, x)
        total_err += (0.5 * np.linalg.norm(y_pred - y)**2)
    return total_err / len(testing_data)

def predict(W, B, x):
    y = x
    for i in range(len(W)):
        y = sig(np.dot(W[i], y) + B[i])
    return y

################################################################################################ helper functions
def sig(x):
    return (1 + np.exp(-x))**(-1)

#Derivative of Sigmoid Function
def sigmoid(x):
    return sig(x) * (1 - sig(x))

# Plot the estimated function vs the original function in one graph
def plot_original_vs_estimate(x, y, y_pred):
    # Sort the value of x in order to plot the function
    x, y, y_pred = zip(*sorted(zip(x, y, y_pred)))
    plt.plot(x, y, label='original function')
    plt.plot(x, y_pred, label='estimated function')
    plt.title('Comparison between original and estimate')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# split the dataset into training and testing
def split_data(X, Y, training_ratio):
    n = len(X)
    mid = floor(n*training_ratio)
    training_X = [x.reshape(1, 1) for x in X[:mid]]
    training_Y = [y.reshape(1, 1) for y in Y[:mid]]
    testing_X = [x.reshape(1, 1) for x in X[mid+1:]]
    testing_Y = [y.reshape(1, 1) for y in Y[mid+1:]]
    training_data = list(zip(training_X, training_Y))
    testing_data = list(zip(testing_X, testing_Y))
    return [training_data, testing_data]

if __name__ == '__main__':
    # Load Data
    mat_contents = sio.loadmat('hw2data_2.mat')
    X = mat_contents['X']
    Y = mat_contents['Y']

    # Set up parameters
    passes = 2000
    eta = 0.005
    split_ratio = 0.95
    numHidden = 500

    # Split dataset
    training_data, testing_data = split_data(X, Y, split_ratio)

    # Train the neural net
    Whidden=np.random.normal(0, 1/np.sqrt(numHidden), (numHidden, 1))
    Bhidden=np.random.randn(numHidden, 1)
    Wout=np.random.normal(0, 1/np.sqrt(numHidden), (1, numHidden))
    Bout=np.random.randn(1, 1)
    [W, B]= train(training_data, testing_data, passes, eta, Whidden, Bhidden, Wout, Bout)

    testing_X = []
    testing_Y = []
    predicted_Y = []

    for x, y in testing_data:
        testing_X.append(x[0][0])
        testing_Y.append(y[0][0])
        predicted_Y.append(predict(W, B, x)[0][0])

    plot_original_vs_estimate(testing_X, testing_Y, predicted_Y)