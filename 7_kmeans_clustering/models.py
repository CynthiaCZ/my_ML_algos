"""
    This is the class file contains helper functions for the Kmeans classifier
"""
import numpy as np
from kmeans import kmeans

class KmeansClassifier(object):
    """
    K-Means Classifier via Iterative Improvement
    """

    def __init__(self, n_clusters = 10, max_iter = 500, threshold = 1e-6):
        """
        Initiate K-Means
        """
        self.k = n_clusters
        self.tol = threshold
        self.max_iter = max_iter
        self.cluster_centers_ = np.array([])

    def train(self, X):
        """
        Compute K-Means clustering on each class label and store the result in self.cluster_centers_
        """
        self.cluster_centers_ = kmeans(X, self.k, self.max_iter, self.tol)

    def predict(self, X, centroid_assignments):
        """
        Predicts the label of each sample in X based on the assigned centroid_assignments
        """
        predictions = np.zeros(len(X))
        for i in range(len(X)):
            distances = np.zeros(self.k)
            for j in range(self.k):
                distances[j] = np.linalg.norm(X[i] - self.cluster_centers_[j])
            centroids = np.argmin(distances)
            predictions[i] = centroid_assignments[centroids]

        return predictions


    def accuracy(self, data, centroid_assignments):
        """
        Compute accuracy of the model when applied to data
        """
        pred = self.predict(data.inputs, centroid_assignments)
        return np.mean(pred == data.labels)
