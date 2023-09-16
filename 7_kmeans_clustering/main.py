"""
    This file contains the main program to read data, run the Kmeans classifier and print results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
import random
from sklearn.model_selection import train_test_split

from models import KmeansClassifier


def plot_Kmeans(model):
    """
       Takes in a pre-trained K-Means classifier model and plots the 10 centroids.
       (only works for the digits.csv dataset)
    """
    if isinstance(model, KmeansClassifier) == False:
        print("Invalid input! Model must be a KmeansClassifier object.")
        return

    cluster_centers = model.cluster_centers_
    fig, ax = plt.subplots(1, len(cluster_centers), figsize=(3, 1))

    for i in range(len(cluster_centers)):
        axi = ax[i]
        center = cluster_centers[i]
        center = np.array(center).reshape(8,8)
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    fig.savefig("./plot_clusters")

def test_Kmeans(model, test_data, centroid_assignments):
    """
       Prints the accuracy of model on test_data, based on the centroid_assignments
    """
    if not np.array_equal(np.array(list(range(10))), np.sort(np.array(centroid_assignments))):
        print("Invalid Input! Centroid assignments must contain all numbers in the range 0-9 (in the order displayed in your plot).")
        return 
    
    accuracy = model.accuracy(test_data, centroid_assignments)
    print("Testing on K-Means Classifier (K = " + str(model.k) + "), the accuracy is {:.2f}%".format(accuracy * 100))
    # accuracy: 80.94%
    return accuracy

def runKMeans():
    '''
        Trains, plots, and tests K-Means classifier on digits.csv dataset.
    '''
    NUM_CLUSTERS = 10 # the ten digits in the digits.csv dataset
    random.seed(1)
    np.random.seed(1)

    # get data
    Dataset = namedtuple('Dataset', ['inputs', 'labels'])
    data = pd.read_csv("./data/digits.csv", header = 0)
    labels = data.values[:, 0]

    # If labels are of type string, convert class names to numeric values
    if isinstance(labels[0], str):
        classes = np.unique(labels)
        class_mapping = dict(zip(classes, range(0, len(classes))))
        labels = np.vectorize(class_mapping.get)(labels)

    # Features columns
    inputs = data.values[:, 1:].astype("float32")

    # Split data into training set and test set with a ratio of 2:1
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size = 0.33)

    all_data = Dataset(inputs, labels)
    train_data = Dataset(train_inputs, train_labels)
    test_data = Dataset(test_inputs, test_labels)
    print("Shape of training data inputs: ", train_data.inputs.shape)
    print("Shape of test data inputs:", test_data.inputs.shape)

    # Train K-Means Classifier
    kmeans_model = KmeansClassifier(NUM_CLUSTERS)
    kmeans_model.train(train_data.inputs)

    # plot the centroids for the 10 digits (0-9).
    plot_Kmeans(kmeans_model)

    # fill out centroid_assignments below based on the visualization of plot_Kmeans
    test_Kmeans(kmeans_model, test_data, centroid_assignments=[9,2,1,4,3,0,8,6,5,7])

def main():
    runKMeans()

if __name__ == '__main__':
    main()
