'''
   This file contains the main program to read data, run the model, and print results
   The model is being applied to the UCI Breast Cancer Wisconsin (Diagnostic) Data,
   and the classification task is to predict whether or not a given patient has breast cancer based on health data
   https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
'''

import numpy as np
import pandas as pd
from models import RegularizedLogisticRegression

def extract():
    X_train = pd.read_csv('./data/X_train.csv',header=None)
    Y_train = pd.read_csv('./data/y_train.csv',header=None)
    X_val = pd.read_csv('./data/X_val.csv',header=None)
    Y_val = pd.read_csv('./data/y_val.csv',header=None)

    Y_train = np.array([i[0] for i in Y_train.values])
    Y_val = np.array([i[0] for i in Y_val.values])

    X_train = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_val = np.append(X_val, np.ones((len(X_val), 1)), axis=1)

    return X_train, X_val, Y_train, Y_val

def main():
    # regularized logistic regression on 70-15-15 split dataset
    X_train, X_val, Y_train, Y_val = extract()
    X_train_val = np.concatenate((X_train, X_val))
    Y_train_val = np.concatenate((Y_train, Y_val))

    RR = RegularizedLogisticRegression()
    RR.train(X_train, Y_train)
    print('Train Accuracy: ' + str(RR.accuracy(X_train, Y_train)))
    print('Validation Accuracy: ' + str(RR.accuracy(X_val, Y_val)))

    # Train Accuracy: ~0.87
    # Validation Accuracy: ~0.84

    lambda_list = [1000, 100, 10, 1, 0.1, 0.01, 0.001]
    train_errors, val_errors = RR.runTrainTestValSplit(lambda_list, X_train, Y_train, X_val, Y_val)

    # k-fold validation with k=3
    k_fold_errors = RR.runKFold(lambda_list, X_train_val, Y_train_val, 3)
    print(lambda_list)
    print(train_errors, val_errors, k_fold_errors)

    # plot errors with respect to lambdas
    RR.plotError(lambda_list, train_errors, val_errors, k_fold_errors)
    # lambda = 1 has the lowest validation error and k-fold error
    # but there are some fluctuations for different random states


if __name__ == '__main__':
    main()
