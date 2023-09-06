'''
   This file contains functions to evaluate models by splitting data into train/test/validation sets,
   and returns lists of training and validation errors with respect to a selection of lambda hyperparameter values.
   This file also contains functions to evaluate models by implementing k-fold cross validation, 
   and returns a list of errors with respect to each value of lambda.
'''

import numpy as np
import matplotlib.pyplot as plt

def sigmoid_function(x):
    return 1.0 / (1.0 + np.exp(-x))

class RegularizedLogisticRegression(object):
    '''
    Regularized logistic regression for binary classification
    The weight vector w is learned by minimizing the regularized loss 
    (log loss plus Tikhonov regularization with a coefficient of lambda)
    '''
    def __init__(self):
        self.learningRate = 0.00001 #tuned parameter
        self.num_epochs = 10000 #tuned parameter
        self.batch_size = 15 #tuned parameter
        self.weights = None
        self.lmbda = 1 #tuned parameter

    def train(self, X, Y):
        '''
        Train the model, using batch stochastic gradient descent
        '''
        # initialize
        self.weights = np.zeros((1, X.shape[1]))
        b = self.batch_size

        # run for fixed number of epochs, shuffle before each epoch
        for k in range(self.num_epochs):
            randomize = np.arange(X.shape[0])
            np.random.shuffle(randomize)
            X = X[randomize]
            Y = Y[randomize]

            for i in range(X.shape[0]//b):
                # for each batch
                X_batch = X[i*b:(i+1)*b]
                Y_batch = Y[i*b:(i+1)*b]
                b_gradient = np.zeros((1, X.shape[1]))

                for x,y in zip(X_batch,Y_batch):
                    b_logits = self.weights @ x.T
                    b_hx = sigmoid_function(b_logits)
                    b_gradient += (b_hx-y)*x + 2*self.lmbda*self.weights

            self.weights -= self.learningRate * b_gradient/len(X_batch)
        
    def predict(self, X):
        '''
        Compute predictions based on the learned weights 
        '''
        logits = self.weights @ X.T
        hx = sigmoid_function(logits)
        prediction = hx>0.5
        return prediction.T

    def accuracy(self,X, Y):
        '''
        Output the accuracy of the trained model
        '''
        prediction = self.predict(X)
        Y = np.reshape(Y,(Y.shape[0],1))
        accuracy = np.sum(prediction == Y)/X.shape[0]
        return accuracy

    def runTrainTestValSplit(self, lambda_list, X_train, Y_train, X_val, Y_val):
        '''
        Given the training and validation data, fit the model with training data and test it with
        respect to each lambda. Record the training error and validation error
        '''
        train_errors = []
        val_errors = []

        for lmbda in lambda_list:
            self.lmbda = lmbda

            self.train(X_train, Y_train)
            train_errors = np.append(train_errors, 1-self.accuracy(X_train, Y_train))
            val_errors = np.append(val_errors, 1-self.accuracy(X_val, Y_val))

        return train_errors, val_errors

    def _kFoldSplitIndices(self, dataset, k):
        '''
        k-fold cross validation: split the indices of the dataset into k groups
        '''
        num_data = dataset.shape[0]
        fold_size = int(num_data / k)
        indices = np.arange(num_data)
        np.random.shuffle(indices)
        indices_split = np.split(indices[:fold_size*k], k)
        return indices_split

    def runKFold(self, lambda_list, X, Y, k = 3):
        '''
        Run k-fold cross validation on X and Y with respect to each lambda
        '''
        k_fold_errors = []
        for lmbda in lambda_list:
            self.lmbda = lmbda
            # Split indices into k groups randomly
            indices_split = self._kFoldSplitIndices(X, k)

            # For each iteration i = 1...k, train the model using lmbda on k−1 folds of data
            # Then test with the i-th fold.
            single_error = []
            for i in range(k):
                test_indices = indices_split[i]
                X_test = X[test_indices]
                Y_test = Y[test_indices]
                X_train = np.delete(X, test_indices, 0)
                Y_train = np.delete(Y, test_indices, 0)

                self.train(X_train,Y_train)
                single_error = np.append(single_error, 1-self.accuracy(X_train, Y_train))

                single_error = np.append(single_error, 1-self.accuracy(X_test, Y_test))

            # Average total errors across folds
            k_fold_errors = np.append(k_fold_errors, np.average(single_error))


        return k_fold_errors

    def plotError(self, lambda_list, train_errors, val_errors, k_fold_errors):
        '''
        Plot the cost function on the training and validation sets,
        and the cost function of k-fold with respect to the regularization parameter lambda
        '''
        plt.figure()
        plt.semilogx(lambda_list, train_errors, label = 'training error')
        plt.semilogx(lambda_list, val_errors, label = 'validation error')
        plt.semilogx(lambda_list, k_fold_errors, label = 'k-fold error')
        plt.xlabel('lambda')
        plt.ylabel('error')
        plt.legend()
        plt.show()