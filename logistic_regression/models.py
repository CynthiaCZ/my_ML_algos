
'''
   This file contains the functions for logistic regression
'''
import numpy as np

def softmax(x):
    '''
    Apply the softmax fuction to an array
    '''
    e = np.exp(x - np.max(x))
    return (e + 1e-6) / (np.sum(e) + 1e-6)

class LogisticRegression:
    '''
    multiclass logistic regression with stochastic gradient descent
    '''
    def __init__(self, n_features, n_classes, batch_size, conv_threshold):

        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.zeros((n_classes, n_features + 1))  # padded for bias
        self.alpha = 0.03 # learning rate
        self.batch_size = batch_size # batch size
        self.conv_threshold = conv_threshold # convergence threashold

    def train(self, X, Y):
        '''
        Trains the model using stochastic gradient descent
        '''
        # initialize
        converge = False
        epoch = 0
        b = self.batch_size
        last_epoch_loss = None

        # stochastic gradient descent 
        while not converge:
            epoch += 1
            print(epoch)

            # shuffle before each epoch
            randomize = np.arange(len(X))
            np.random.shuffle(randomize)
            X = X[randomize]
            Y = Y[randomize]

            for i in range(len(X)//b):
                # for each batch
                X_batch = X[i*b:(i+1)*b]
                Y_batch = Y[i*b:(i+1)*b]
                b_gradient = np.zeros(self.weights.shape)

                for x,y in zip(X_batch,Y_batch):
                    for j in range(self.n_classes):
                        # calculate gradient
                        if y==j:
                            b_gradient[j] += (softmax(self.weights @ x)[j]-1)*x
                        else:
                            b_gradient[j] += (softmax(self.weights @ x)[j])*x
                # update weights
                self.weights -= self.alpha*b_gradient/len(X_batch)
            this_epoch_loss = self.loss(X,Y)
            print("loss: ", this_epoch_loss)

            # Check if converged
            if last_epoch_loss is not None and abs(this_epoch_loss - last_epoch_loss) < self.conv_threshold:
                converge = True
            last_epoch_loss = this_epoch_loss
        return epoch

    def loss(self, X, Y):
        '''
        Returns the total log loss divided by the number of examples.
        '''
        logits = self.weights @ X.T
        # for each data point x, calculates the probabilities that x belongs to each class
        probs = np.apply_along_axis(softmax, 0, logits)
        p_true = probs[Y, np.arange(X.shape[0])]
        loss = np.log(p_true)
        avg_loss = -np.sum(loss) / X.shape[0]
        return avg_loss

    def predict(self, X):
        '''
        Returns predictions based on the learned weights
        '''
        logits = self.weights @ X.T
        # for each data point x, get the class with the highest probability
        probs = np.apply_along_axis(softmax, 0, logits)
        prediction = np.argmax(probs, axis=0)
        return prediction

    def accuracy(self, X, Y):
        '''
        Returns the accuracy of the trained model
        '''
        prediction = self.predict(X) 
        accuracy = sum(prediction == Y)/X.shape[0]
        return accuracy