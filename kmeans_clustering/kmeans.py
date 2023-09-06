"""
    This file contains helper functions required by the K-means method via iterative improvement
"""
import numpy as np
from random import sample

def init_centroids(k, inputs):
    """
    Selects k random rows from inputs and returns them as the chosen centroids
    """
    row_indeices = sample(range(inputs.shape[0]), k)
    return inputs[row_indeices]


def assign_step(inputs, centroids):
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance
    """
    distances = np.zeros([len(inputs),len(centroids)])
    for i in range(len(inputs)):
        for j in range(len(centroids)):
            distances[i,j] = np.linalg.norm(inputs[i] - centroids[j])
    return np.argmin(distances, axis=1)


def update_step(inputs, indices, k):
    """
    Computes the centroid for each cluster
    """
    centroids = np.zeros([k, inputs.shape[1]])
    for i in range(k):
        centroids[i] = np.sum(inputs[indices==i],axis=0) / sum(indices==i)
    return centroids


def kmeans(inputs, k, max_iter, tol):
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement
    """
    old_centroids = init_centroids(k, inputs)
    converge = False
    iter = 0
    
    # keep optimizing the clusters until either
    #   1. the number of iterations reaches max_iter
    #   2. the ratio of the norm of the difference between centroids and the norm of the original centroids 
    #       reaches the tolerance 
    while converge == False:
        indices = assign_step(inputs, old_centroids)
        new_centroids = update_step(inputs, indices, k)
        iter += 1
        ratio = np.linalg.norm(new_centroids - old_centroids) / np.linalg.norm(old_centroids)
        
        # check for convergence
        if iter > max_iter or ratio < tol:
            converge = True

        old_centroids = new_centroids
    return new_centroids
