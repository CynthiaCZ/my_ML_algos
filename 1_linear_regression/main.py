"""
   This file contains the main program to read data, run the model, and print results
   The model is being applied to a dataset on wine quality: https://archive.ics.uci.edu/ml/datasets/Wine+Quality.
   A wine quality rating (out of 10) is being predicted based on 11 attributes
"""

import numpy as np
import random
from sklearn.model_selection import train_test_split
from models import LinearRegression

WINE_FILE_PATH = './data/wine.txt'

def import_wine(filepath, test_size=0.2):
    '''
        Import the wine dataset and split train and test datasets
    '''
    # Load
    data = np.loadtxt(filepath, skiprows=1)
    X, Y = data[:, 1:], data[:, 0]

    # Normalize
    X = (X-np.mean(X, axis=0))/np.std(X, axis=0)

    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    return X_train, X_test, Y_train, Y_test


def test_linreg():
    '''
        Tests and outputs training and testing loss of LinearRegression
    '''
    X_train, X_test, Y_train, Y_test = import_wine(WINE_FILE_PATH)

    num_features = X_train.shape[1]

    # Padding the inputs with a bias
    X_train_b = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_test_b = np.append(X_test, np.ones((len(X_test), 1)), axis=1)

    #### Matrix Inversion ######
    print('---- LINEAR REGRESSION w/ Matrix Inversion ---')
    solver_model = LinearRegression(num_features)
    solver_model.train(X_train_b, Y_train)
    print('Average Training MSE:', solver_model.average_loss(X_train_b, Y_train))
    print('Average Testing MSE:', solver_model.average_loss(X_test_b, Y_test))
    # training MSE = 0.540
    # testing MSE = 0.659

def main():
    random.seed(0)
    np.random.seed(0)
    test_linreg()

if __name__ == "__main__":
    main()
