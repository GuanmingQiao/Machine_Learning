# -*- coding: utf-8 -*-
"""
Spyder Editor

HW1 P6 Decision Tree.
"""

from __future__ import print_function
import scipy.io
from random import shuffle
from math import floor
import numpy as np
import matplotlib.pyplot as plt

# recursive class node
class Node:
     def __init__(self, index, value, children):
         self.index =index
         self.value = value
         self.children = children
     def deleteChildren(self):
         self.children = None
     def getChildren(self):
         return self.children
     def getIndex(self):
         return self.index
     def getValue(self):
         return self.value
     def printNode(self):
         print({self.index, self.value}, sep=' ', end='')
     def getLeft(self):
         return self.children[0]
     def getRight(self):
         return self.children[1]
     def setChildren(self, children):
         self.children = children

# load and preprocess data
def loadDataset(filename, split):
    mat = scipy.io.loadmat(filename)
    data_X = mat['X']
    data_Y = mat['Y']
    data = []
    for i in range(data_X.shape[0]):
        # split the data into 1's and 0's
        bins = np.array([0.0, 0.1, 127.5, 255.0])
        data_X[i] = np.digitize(data_X[i], bins)
        list_X = data_X[i].tolist()
        list_X.append(data_Y[i][0])
        data.append(list_X)
    shuffle(data)
    n = len(data)
    mid = floor(n*split)
    training_data = data[:mid]
    testing_data = data[mid+1:]
    return [training_data, testing_data]

# Split a dataset based on a feature index and a feature value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
# groups: left, right. labels: 1-9
def gini_index(splits):
    n = float(sum([len(group) for group in splits]))
    gini = 0.0
    for group in splits:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for i in range(10):
            votes = []
            for row in group:
                votes.append(row[-1])
            p = votes.count(i) / size
            score += p * p
        gini += (1.0 - score) * (size / n)
    return gini

def contains(visited, target):
    for num in visited:
        if num == target:
            return 1
    visited.append(target)
    return 0

def get_split(dataset):
    b_index, b_value, b_score, b_groups = 100000, 100000, 100000, None
    for index in range(len(dataset[0])-1):
        visited = []
        for row in dataset:
            if(contains(visited, row[index]) == 1):
                continue
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups)
            if gini < b_score:
                #print(gini, dataset.index(row),index)
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return Node(b_index, b_value, b_groups)

def split(node, max_depth, depth):
    #print('Current level: ', depth)
    [left, right] = node.getChildren()
    node.deleteChildren()
    # if no split
    if not left or not right:
        outcomes = [row[-1] for row in (left + right)]
        leftChild = max(set(outcomes), key=outcomes.count)
        rightChild = max(set(outcomes), key=outcomes.count)
        node.setChildren([leftChild, rightChild])
        return
	# if max depth
    if depth >= max_depth:
        outcomes = [row[-1] for row in (left)]
        leftChild = max(set(outcomes), key=outcomes.count)
        outcomes = [row[-1] for row in (right)]
        rightChild = max(set(outcomes), key=outcomes.count)
        node.setChildren([leftChild, rightChild])
        return
	#  left child
    leftChild = get_split(left)
    #  right child
    rightChild = get_split(right)
    node.setChildren([leftChild, rightChild])
    split(node.getLeft(), max_depth, depth+1)
    split(node.getRight(), max_depth, depth+1)

# Make a prediction with a decision tree
def predict(node, row):
	if row[node.getIndex()] < node.getValue():
		if isinstance(node.getLeft(), Node):
			return predict(node.getLeft(), row)
		else:
			return node.getLeft()
	else:
		if isinstance(node.getRight(), Node):
			return predict(node.getRight(), row)
		else:
			return node.getRight()

# evaluate success rate of a decision tree parameter
def evaluate(train, data, max_depth):
    tree = build_tree(train, max_depth)
    results = []
    for dataset in data:
        success = 0
        for row in dataset:
            if (row[-1] == predict(tree, row)):
                success = success + 1
        results.append(1 - success/len(dataset))
    return results

# build a decision tree
def build_tree(train, max_depth):
	root = get_split(train)
	split(root, max_depth, 1)
	return root

# plot the results
def plot(train, test, depths):
    training_errors = []
    testing_errors = []
    for depth in depths:
        results = evaluate(train, [test, train], depth)
        testing_errors.append(results[0])
        print('depth: ', depth, 'testing error: ', results[0])
        training_errors.append(results[1])
        print('depth: ', depth, 'training error: ', results[1])
    plt.plot(depths, training_errors, 'r--', label='training error')
    plt.plot(depths, testing_errors, 'b--', label='testing error')
    plt.title('maximum depth VS error rate of decision tree')
    plt.xlabel('depth')
    plt.ylabel('error rate')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    data = loadDataset('hw1data.mat', 0.9)
    training_data = data[0]
    testing_data = data[1]
    plot(training_data, testing_data, [4, 6, 8, 12, 14])