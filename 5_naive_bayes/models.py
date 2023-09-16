'''
   This file contains functions to implement a Naive Bayes model
   as well as different methods to gauge the trained model's fairness
'''
import numpy as np

class NaiveBayes(object):
    """ Bernoulli Naive Bayes model
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.attr_dist = None
        self.label_priors = None

    def train(self, X_train, y_train):
        """ 
        Trains the model using maximum likelihood estimation
        """
        n_examples = len(X_train)
        n_attributes = len(X_train[0])

        # Priors, 1st Laplace smoothing
        self.label_priors = np.zeros(2)
        self.label_priors[0] = (n_examples-sum(y_train)+1)/(n_examples+2) 
        self.label_priors[1] = (sum(y_train)+1)/(n_examples+2) 

        # Attributes, 2nd Laplace smoothing
        self.attr_dist = np.zeros((n_attributes, 2))
        class0 = X_train[y_train==0]
        class1 = X_train[y_train==1]
        for attr in range(n_attributes):
            self.attr_dist[attr,0] = (sum(class0[:,attr])+1)/(len(class0)+2)
            self.attr_dist[attr,1] = (sum(class1[:,attr])+1)/(len(class1)+2)
        return self.attr_dist, self.label_priors

    def predict(self, inputs):
        """ 
        Outputs a predicted label for each input in inputs.
        Converted to log space to avoid overflow/underflow
        """
        predictions = np.zeros(len(inputs))
        for i in range(len(inputs)):
            input = inputs[i]
            prob_table = np.zeros(np.shape(self.attr_dist))
            prob_table[input==1] = self.attr_dist[input==1]
            prob_table[input==0] = 1-self.attr_dist[input==0]
            log_prob_table = np.log(prob_table)
            prob = np.exp(log_prob_table.sum(axis=0)) * self.label_priors
            predictions[i] = np.argmax(prob)
        return predictions

    def accuracy(self, X_test, y_test):
        """ 
        Outputs the accuracy of the trained model on dataset
        """
        predictions = self.predict(X_test)
        accuracy = sum(predictions==y_test)/len(y_test)
        return accuracy

    def print_fairness(self, X_test, y_test, x_sens):
        """ 
        Prints measures of the trained model's fairness on a given dataset

        For all of these measures, x_sens == 1 corresponds to the "privileged"
        class, and x_sens == 0 corresponds to the "disadvantaged" class
        y == 1 corresponds to "good" credit
        """
        predictions = self.predict(X_test)

        # Disparate Impact (80% rule):
        # the data has disparate impact if:
        # P[Y^ = 1 | S != 1] / P[Y^ = 1 | S = 1] <= (t = 0.8). 
        di = np.mean(predictions[np.where(x_sens==0)])/np.mean(predictions[np.where(x_sens==1)])
        print("Disparate impact: " + str(di))

        # Group-conditioned error rates
        # False positives/negatives conditioned on group
        pred_priv = predictions[np.where(x_sens==1)]
        pred_unpr = predictions[np.where(x_sens==0)]
        y_priv = y_test[np.where(x_sens==1)]
        y_unpr = y_test[np.where(x_sens==0)]

        # s-TPR (true positive rate) = P[Y^=1|Y=1,S=s]
        priv_tpr = np.sum(np.logical_and(pred_priv == 1, y_priv == 1))/np.sum(y_priv)
        unpr_tpr = np.sum(np.logical_and(pred_unpr == 1, y_unpr == 1))/np.sum(y_unpr)

        # s-TNR (true negative rate) = P[Y^=0|Y=0,S=s]
        priv_tnr = np.sum(np.logical_and(pred_priv == 0, y_priv == 0))/(len(y_priv) - np.sum(y_priv))
        unpr_tnr = np.sum(np.logical_and(pred_unpr == 0, y_unpr == 0))/(len(y_unpr) - np.sum(y_unpr))

        # s-FPR (false positive rate) = P[Y^=1|Y=0,S=s]
        priv_fpr = 1 - priv_tnr 
        unpr_fpr = 1 - unpr_tnr 

        # s-FNR (false negative rate) = P[Y^=0|Y=1,S=s]
        priv_fnr = 1 - priv_tpr 
        unpr_fnr = 1 - unpr_tpr

        print("FPR (priv, unpriv): " + str(priv_fpr) + ", " + str(unpr_fpr))
        print("FNR (priv, unpriv): " + str(priv_fnr) + ", " + str(unpr_fnr))

        return predictions
