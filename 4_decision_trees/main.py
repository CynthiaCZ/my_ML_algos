'''
This file contains the main program to read data, run the model, and print results
   The model is being applied to 
   1. Spambase Dataset: the goal is to train a model that can classify whether an email is spam or not
    from features attributes such as the frequency of certain words and the amount of capital letters in a given message
    https://archive.ics.uci.edu/dataset/94/spambase
   2. Chess Dataset
    Each row of the chess.csv dataset contains 36 features, which represent the current state of the chess board
    The task is to use the Decision Trees to classify whether or not it is possible for white to win the game
    https://archive.ics.uci.edu/dataset/21/chess+king+rook+vs+king+knight
'''

import random
import numpy as np
import matplotlib.pyplot as plt

from get_data import get_data
from models import DecisionTree, node_score_error, node_score_entropy, node_score_gini

def explore_dataset(filename, class_name):
    train_data, validation_data, test_data = get_data(filename, class_name)
    # For each measure of gain (training error, entropy, gini):
    #   (a) Print average training loss (not-pruned)
    #   (b) Print average test loss (not-pruned)
    #   (c) Print average training loss (pruned)
    #   (d) Print average test loss (pruned)
    
    decision_entropy = DecisionTree(train_data)
    training_loss = decision_entropy.loss(train_data)
    print('entropy: training loss not pruned', training_loss)
    test_loss = decision_entropy.loss(test_data)
    print('entropy: test loss not pruned', test_loss)

    decision_entropy_pruned = DecisionTree(train_data,validation_data=validation_data)
    training_loss = decision_entropy_pruned.loss(train_data)
    print('entropy: training loss pruned', training_loss)
    test_loss = decision_entropy_pruned.loss(test_data)
    print('entropy: test loss pruned', test_loss)

    decision_gini = DecisionTree(train_data, gain_function=node_score_gini)
    training_loss = decision_gini.loss(train_data)
    print('gini: training loss not pruned', training_loss)
    test_loss = decision_gini.loss(test_data)
    print('gini: test loss not pruned', test_loss)

    decision_gini_pruned = DecisionTree(train_data,validation_data=validation_data, gain_function=node_score_gini)
    training_loss = decision_gini_pruned.loss(train_data)
    print('gini: training loss pruned', training_loss)
    test_loss = decision_gini_pruned.loss(test_data)
    print('gini: test loss pruned', test_loss)

    decision_error = DecisionTree(train_data, gain_function=node_score_error)
    training_loss = decision_error.loss(train_data)
    print('decision error: training loss not pruned', training_loss)
    test_loss = decision_error.loss(test_data)
    print('decision error: test loss not pruned', test_loss)

    decision_error_pruned = DecisionTree(train_data,validation_data=validation_data, gain_function=node_score_error)
    training_loss = decision_error_pruned.loss(train_data)
    print('decision error: training loss pruned', training_loss)
    test_loss = decision_error_pruned.loss(test_data)
    print('decision error: test loss pruned', test_loss)

    # plot loss at different max depths
    depth_vector = list(range(1, 16))
    loss_vec = np.zeros(15)
    for i in range(15):
        decision_plot = DecisionTree(train_data, max_depth=i+1)
        loss_vec[i] = decision_plot.loss(train_data)
    plt.plot(depth_vector,loss_vec)
    plt.ylabel('loss')
    plt.xlabel('max depth')
    plt.title('loss with different max')
    plt.show()

def main():
    random.seed(0)
    np.random.seed(0)

    explore_dataset('data/chess.csv', 'won')
    explore_dataset('data/spam.csv', '1')

main()
