"""
   This file contains the main program to read data, run the model, and print results
   The model is being applied to the UCI Census Income data from 1994
   url: https://archive.ics.uci.edu/ml/datasets/Census+Income
   Education levels (3 levels) of individuals are being predicted based on the attributes from the census
"""

import numpy as np
import random
from sklearn.model_selection import train_test_split
from models import LogisticRegression

ROOT_DIR_PREFIX = './data/'
DATA_FILE_NAME = 'normalized_data.csv'
CENSUS_FILE_PATH = ROOT_DIR_PREFIX + DATA_FILE_NAME

NUM_CLASSES = 3
BATCH_SIZE = 5  #tuned parameter
CONV_THRESHOLD = 1e-4 #tuned parameter

def import_census(file_path):
    '''
        Import the census dataset and split train and test datasets
    '''
    data = np.genfromtxt(file_path, delimiter=',', skip_header=False)
    X = data[:, :-1]
    Y = data[:, -1].astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    return X_train, Y_train, X_test, Y_test

def test_logreg():
    X_train, Y_train, X_test, Y_test = import_census(CENSUS_FILE_PATH)
    num_features = X_train.shape[1]

    # Add a bias
    X_train_b = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_test_b = np.append(X_test, np.ones((len(X_test), 1)), axis=1)

    ### Logistic Regression ###
    model = LogisticRegression(num_features, NUM_CLASSES, BATCH_SIZE, CONV_THRESHOLD)
    num_epochs = model.train(X_train_b, Y_train)
    acc = model.accuracy(X_test_b, Y_test) * 100
    print("Test Accuracy: {:.1f}%".format(acc))
    print("Number of Epochs: " + str(num_epochs))

    return acc

# Test Accuracy: 88.2%
# Number of Epochs: 33

def main():
    random.seed(0)
    np.random.seed(0)

    test_logreg()

if __name__ == "__main__":
    main()
