'''
   This file contains the functions for linear regression
'''
import numpy as np

def squared_error(predictions, Y):
    '''
    Calculate sum squared/L2 loss from true values and predictions
    '''
    sq_loss = np.sum(np.square(Y - predictions))
    return sq_loss

class LinearRegression:
    '''
    LinearRegression model that minimizes squared error using matrix inversion
    '''
    def __init__(self, n_features):
        self.n_features = n_features + 1  # padded for bias
        self.weights = np.zeros(n_features + 1)

    def train(self, X, Y):
        '''
        Trains the LinearRegression model by finding the optimal set of weights using matrix inversion
        '''
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ Y
        return self.weights

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X
        '''
        hx = X @ self.weights
        return hx

    def loss(self, X, Y):
        '''
        Returns the total squared error
        '''
        predictions = self.predict(X)
        return squared_error(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error
        MSE = Total squared error/# of examples
        '''
        return self.loss(X, Y)/X.shape[0]
