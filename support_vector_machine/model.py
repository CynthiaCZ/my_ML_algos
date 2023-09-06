'''
   This file contains functions to implement support vector machine (SVM) model
   to solve binary classification problems
   A Python library for solving quadratic problems, quadprog, is used as an optimizer
   instead of gradient-based methods
'''

import numpy as np
from qp import solve_QP


def linear_kernel(xi, xj):
    """
    linear kernel (regular dot product)
    """
    K = np.dot(xi,xj)
    return K


def rbf_kernel(xi, xj, gamma=0.1):
    """
    radial basis function kernel (RBF) with hyperparameter gamma
    """
    K = np.exp(-gamma*(np.linalg.norm(xi-xj))**2)
    return K


def polynomial_kernel(xi, xj, c=2, d=2):
    """
    polynomial kernel
    c: mean of the polynomial kernel
    d: exponent of the polynomial
    """
    K = (np.dot(xi,xj) + c)**d
    return K


class SVM(object):

    def __init__(self, kernel_func=linear_kernel, lambda_param=.1):
        self.kernel_func = kernel_func
        self.lambda_param = lambda_param

    def train(self, inputs, labels):
        """
        Train the model with the input data (inputs and labels),
        Find the coefficients and constaints for the quadratic program and calculate the alphas
        """
        self.train_inputs = inputs
        self.train_labels = labels

        # constructing QP variables
        G = self._get_gram_matrix()
        Q, c = self._objective_function(G)
        A, b = self._inequality_constraint(G)

        # solve quadratic programs
        self.alpha = solve_QP(Q, c, A, b)[:self.train_inputs.shape[0]]

    def _get_gram_matrix(self):
        """
        Generate the Gram matrix G for the training data stored in self.train_inputs
        Element i, j of the matrix is K(x_i, x_j), where K is the kernel function
        """
        m = len(self.train_inputs)
        G = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                xi = self.train_inputs[i]
                xj = self.train_inputs[j]
                G[i,j] = self.kernel_func(xi,xj)
        return G

    def _objective_function(self, G):
        """
        Generate the coefficients on the variables in the objective function for the
        SVM quadratic program.
        The objective is to minimize (1/2)x^T Q x + c^T x
        """
        m = len(self.train_inputs)
        Q = np.zeros((2*m,2*m))
        Q[:m,:m]=G
        Q = Q*2*self.lambda_param

        c = np.zeros(2*m)
        c[m:]=1/m
        return Q, c

    def _inequality_constraint(self, G):
        """
        Generate the inequality constraints for the SVM quadratic program
        The constraints will be enforced so that Ax <= b
        """
        # x is the concatenation of all the alphas and all the xi's and has length of 2m
        m = len(self.train_inputs)
        b = np.zeros(2*m)
        b[m:]=-1

        # construct the four quadrants of matrix A
        A11 = np.zeros((m,m))
        A12 = -1*np.eye(m)
        A21 = -self.train_labels*G*np.eye(m)
        A22 = -1*np.eye(m)

        # concatenate them together
        A1 = np.concatenate((A11, A12), axis=1)
        A2 = np.concatenate((A21, A22), axis=1)
        A = np.concatenate((A1, A2), axis=0)
        return A, b 

    def predict(self, inputs):
        """
        Generate predictions given input using the kernal function
        """
        predictions = np.zeros(len(inputs))
        for i in range(len(inputs)):
            x = inputs[i]
            aK = np.zeros(len(self.train_inputs))
            for j in range(len(self.train_inputs)):
                xj = self.train_inputs[j]
                aK[j] = self.alpha[j]*self.kernel_func(xj,x)
            if np.sum(aK) < 0:
                predictions[i] = -1
            else:
                predictions[i] = 1

        return predictions

    def accuracy(self, inputs, labels):
        """
        Calculate the accuracy of the classifer given inputs and their true labels
        """
        predictions = self.predict(inputs)
        accuracy = sum(labels == predictions)/len(labels)
        return accuracy
