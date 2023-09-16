"""
   This file contains the main program to visualize different kernal functions on two fake datasets
"""

import csv
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model import SVM, linear_kernel, polynomial_kernel, rbf_kernel


def read_data(file_name):
    """
    Reads the data from the input file and splits it into normalized inputs and labels
    """
    inputs, labels, classes = [], [], set()
    with open(file_name) as f:
        positive_label = None
        reader = csv.reader(f)
        for row in reader:
            example = np.array(row)
            classes.add(example[-1])
            # our datasets all start with a True example
            if positive_label is None:
                positive_label = example[-1]
            # converting data points to labels of [-1, 1]
            label = 1 if example[-1] == positive_label else -1
            row.pop()
            labels.append(label)
            inputs.append([float(val) for val in row])

    if len(classes) > 2:
        print('Only binary classification tasks are supported.')
        exit()

    inputs = np.array(inputs)
    labels = np.array(labels)

    # Normalize the feature values
    for j in range(inputs.shape[1]):
        col = inputs[:,j]
        mu = np.mean(col)
        sigma = np.std(col)
        if sigma == 0: sigma = 1
        inputs[:,j] = 1/sigma * (col - mu)

    return inputs, labels

def main():
    """
    Reads in the data, trains an SVM model and outputs the training and
    testing accuracy of the model on the dataset
    """
    ###### fake data 1 linear kernel ######
    inputs, labels = read_data('data/fake-data1.csv')

    # decision boundary
    svm_model = SVM(kernel_func=linear_kernel, lambda_param=1.0 / (2*inputs.shape[0]))
    svm_model.train(inputs, labels)

    xx, yy = np.meshgrid(np.linspace(inputs.min(), inputs.max(), 100), 
                        np.linspace(inputs.min(), inputs.max(), 100))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    boundary = np.zeros(len(X_grid))
    for i in range(len(X_grid)):
        x = X_grid[i]
        for j in range(len(inputs)):
            xj = inputs[j]
            boundary[i] += svm_model.alpha[j]*svm_model.kernel_func(xj, x)
            
    zz = boundary.reshape(xx.shape)

    # plot
    plt.figure()
    plt.scatter(inputs[:, 0], inputs[:, 1], c=labels)
    plt.contour(xx, yy, zz, levels=[0], colors='k')
    plt.savefig("./plots/linear1.png")


    ###### fake data 1 RBF kernel ######
    # decision boundary
    svm_model = SVM(kernel_func=rbf_kernel, lambda_param=1.0 / (2*inputs.shape[0]))
    svm_model.train(inputs, labels)

    xx, yy = np.meshgrid(np.linspace(inputs.min(), inputs.max(), 100), 
                        np.linspace(inputs.min(), inputs.max(), 100))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    boundary = np.zeros(len(X_grid))
    for i in range(len(X_grid)):
        x = X_grid[i]
        for j in range(len(inputs)):
            xj = inputs[j]
            boundary[i] += svm_model.alpha[j]*svm_model.kernel_func(xj, x)
            
    zz = boundary.reshape(xx.shape)

    # plot
    plt.figure()
    plt.scatter(inputs[:, 0], inputs[:, 1], c=labels)
    plt.contour(xx, yy, zz, levels=[0], colors='k')
    plt.savefig("./plots/rbf1.png")

    ###### fake data 1 polynomial kernel ######
    # decision boundary
    svm_model = SVM(kernel_func=polynomial_kernel, lambda_param=1.0 / (2*inputs.shape[0]))
    svm_model.train(inputs, labels)

    xx, yy = np.meshgrid(np.linspace(inputs.min(), inputs.max(), 100), 
                        np.linspace(inputs.min(), inputs.max(), 100))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    boundary = np.zeros(len(X_grid))
    for i in range(len(X_grid)):
        x = X_grid[i]
        for j in range(len(inputs)):
            xj = inputs[j]
            boundary[i] += svm_model.alpha[j]*svm_model.kernel_func(xj, x)
            
    zz = boundary.reshape(xx.shape)

    # plot
    plt.figure()
    plt.scatter(inputs[:, 0], inputs[:, 1], c=labels)
    plt.contour(xx, yy, zz, levels=[0], colors='k')
    plt.savefig("./plots/polynomial1.png")

    ###### fake data 2 linear kernel ######
    inputs, labels = read_data('data/fake-data2.csv')

    # decision boundary
    svm_model = SVM(kernel_func=linear_kernel, lambda_param=1.0 / (2*inputs.shape[0]))
    svm_model.train(inputs, labels)

    xx, yy = np.meshgrid(np.linspace(inputs.min(), inputs.max(), 100), 
                        np.linspace(inputs.min(), inputs.max(), 100))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    boundary = np.zeros(len(X_grid))
    for i in range(len(X_grid)):
        x = X_grid[i]
        for j in range(len(inputs)):
            xj = inputs[j]
            boundary[i] += svm_model.alpha[j]*svm_model.kernel_func(xj, x)
            
    zz = boundary.reshape(xx.shape)

    # plot
    plt.figure()
    plt.scatter(inputs[:, 0], inputs[:, 1], c=labels)
    plt.contour(xx, yy, zz, levels=[0], colors='k')
    plt.savefig("./plots/linear2.png")


    ###### fake data 2 RBF kernel ######
    # decision boundary
    rbf_with_gamma = lambda x, y: rbf_kernel(x, y, 1.0/inputs.shape[1])
    # default gamma will not 100% train accuracy
    svm_model = SVM(kernel_func=rbf_with_gamma, lambda_param=1.0 / (2*inputs.shape[0]))
    svm_model.train(inputs, labels)

    xx, yy = np.meshgrid(np.linspace(inputs.min(), inputs.max(), 100), 
                        np.linspace(inputs.min(), inputs.max(), 100))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    boundary = np.zeros(len(X_grid))
    for i in range(len(X_grid)):
        x = X_grid[i]
        for j in range(len(inputs)):
            xj = inputs[j]
            boundary[i] += svm_model.alpha[j]*svm_model.kernel_func(xj, x)
            
    zz = boundary.reshape(xx.shape)

    # plot
    plt.figure()
    plt.scatter(inputs[:, 0], inputs[:, 1], c=labels)
    plt.contour(xx, yy, zz, levels=[0], colors='k')
    plt.savefig("./plots/rbf2.png")


    ###### fake data 2 polynomial kernel ######
    # decision boundary
    svm_model = SVM(kernel_func=polynomial_kernel, lambda_param=1.0 / (2*inputs.shape[0]))
    svm_model.train(inputs, labels)

    xx, yy = np.meshgrid(np.linspace(inputs.min(), inputs.max(), 100), 
                        np.linspace(inputs.min(), inputs.max(), 100))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    boundary = np.zeros(len(X_grid))
    for i in range(len(X_grid)):
        x = X_grid[i]
        for j in range(len(inputs)):
            xj = inputs[j]
            boundary[i] += svm_model.alpha[j]*svm_model.kernel_func(xj, x)
            
    zz = boundary.reshape(xx.shape)

    # plot
    plt.figure()
    plt.scatter(inputs[:, 0], inputs[:, 1], c=labels)
    plt.contour(xx, yy, zz, levels=[0], colors='k')
    plt.savefig("./plots/polynomial2.png")

if __name__ == '__main__':
    main()
