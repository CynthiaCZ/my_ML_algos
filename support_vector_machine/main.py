"""
   This file contains the main program to read data, run the model, and print results
   The model is being applied to the Spambase Dataset
   The goal is to train a model that can classify whether an email is spam or not
   from features attributes such as the frequency of certain words and the amount of capital letters in a given message
   https://archive.ics.uci.edu/dataset/94/spambase
"""

import csv
import random
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model import SVM, linear_kernel, polynomial_kernel, rbf_kernel


def test_svm(train_data, test_data, kernel_func=linear_kernel, lambda_param=.1):
    """
    Create an SVM classifier with a specificied kernel_func, train it with
    train_data and print the accuracy of model on test_data
    """
    svm_model = SVM(kernel_func=kernel_func, lambda_param=lambda_param)
    svm_model.train(train_data.inputs, train_data.labels)
    train_accuracy = svm_model.accuracy(train_data.inputs, train_data.labels)
    test_accuracy = svm_model.accuracy(test_data.inputs, test_data.labels)
    if not (train_accuracy is None):
        print('Train accuracy: ', round(train_accuracy * 100, 2), '%')
    if not (test_accuracy is None):
        print('Test accuracy:', round(test_accuracy * 100,2), '%')
    return train_accuracy,test_accuracy

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
    random.seed(0)
    np.random.seed(0)  

    svm_dataset = 'spambase'    
    Dataset = namedtuple('Dataset', ['inputs', 'labels'])
    filename = './data/' + svm_dataset+ '.csv'
    
    # Read data
    inputs, labels = read_data(filename)
    
    print('================ ' + svm_dataset.swapcase() + ' ================')

    # Split data into training set and test set with a ratio of 4:1
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.20)

    train_data = Dataset(train_inputs, train_labels)
    test_data = Dataset(test_inputs[:], test_labels[:])

    print("Shape of training data inputs: ", train_data.inputs.shape)
    print("Shape of test data inputs:", test_data.inputs.shape)

    n = train_data.inputs.shape[0]
    m = train_data.inputs.shape[1]

    # Set lambda parameter
    lambda_param = 1.0 / (2*n)

    print('================ Linear kernel  =================')
    test_svm(train_data, test_data, kernel_func=linear_kernel, lambda_param=lambda_param)
    # Train accuracy:  90.31 %
    # Test accuracy: 87.5 %  

    print('================ RBF kernel =================')
    # Set gamma to 1/m to match the behavior of sklearn's implementation
    rbf_with_gamma = lambda x, y: rbf_kernel(x, y, 1.0/m)
    test_svm(train_data, test_data, kernel_func=rbf_with_gamma, lambda_param=lambda_param)
    # Train accuracy:  95.0 %
    # Test accuracy: 90.0 %

    # Plot the differences in training and testing error as the hyperparameter gamma changes for the RBF kernel
    gamma_list = [1000/n, 100/n, 10/n, 1/n, 0.1/n, 0.01/n]

    train_accuracy = np.zeros(len(gamma_list))
    test_accuracy = np.zeros(len(gamma_list))
    for i in range(len(gamma_list)):
        gamma = gamma_list[i]
        rbf_with_gamma = lambda x, y: rbf_kernel(x, y, gamma)
        train_accuracy[i], test_accuracy[i] = test_svm(train_data, test_data, 
                                                    kernel_func=rbf_with_gamma, lambda_param=lambda_param)
    plt.figure()
    plt.semilogx(gamma_list, train_accuracy, label = 'training accuracy')
    plt.semilogx(gamma_list, test_accuracy, label = 'test accuracy')
    plt.xlabel('gamma')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig("./plots/RBF_accuracy.png")

    print('============= Polynomial kernel =============')
    test_svm(train_data, test_data, kernel_func=polynomial_kernel, lambda_param=lambda_param)
    # Train accuracy:  92.81 %
    # Test accuracy: 90.0 %

    # Plot the differences in training and testing error as the hyperparameter d changes for the polynomial kernel
    d_list = [0,1,2,3,4]
    train_accuracy = np.zeros(5)
    test_accuracy = np.zeros(5)
    for i in range(5):
        d = d_list[i]
        polynomial_with_d = lambda x, y: polynomial_kernel(x, y, c=2, d=d)
        train_accuracy[i], test_accuracy[i] = test_svm(train_data, test_data, 
                                                    kernel_func=polynomial_with_d, lambda_param=lambda_param)
    plt.figure()
    plt.plot(d_list, train_accuracy, label = 'training accuracy')
    plt.plot(d_list, test_accuracy, label = 'test accuracy')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig("./plots/polynomial_accuracy.png")


if __name__ == '__main__':
    main()
