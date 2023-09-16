'''
   This file contains functions to implement decision trees for binary classification problems
   Including recursively splitting and pruning the trees
'''
import numpy as np
import random
import copy
import math

# functions for three types of node scores 
def node_score_error(prob):
    '''
        Calculated using the train error of the subdataset
        C(p) = min{p, 1-p}
    '''
    error = min({prob, 1-prob})
    return error

def node_score_entropy(prob):
    '''
        Calculated using the entropy of the subdataset
        For dataset with 2 classes, C(p) = -p * log(p) - (1-p) * log(1-p)
    '''
    if prob < 1 and prob > 0:
        entropy = -prob * math.log(prob) - (1-prob) * math.log(1-prob)
    else: 
        entropy = 0
    return entropy

def node_score_gini(prob):
    '''
        Calculated using the gini index of the subdataset
        For dataset with 2 classes, C(p) = 2 * p * (1-p)
    '''
    gini = 2 * prob * (1-prob)
    return gini



class Node:
    '''
    Construct the tree structure
    '''
    def __init__(self, left=None, right=None, depth=0, index_split_on=0, isleaf=False, label=1):
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on
        self.isleaf = isleaf
        self.label = label


class DecisionTree:
    def __init__(self, data, validation_data=None, gain_function=node_score_entropy, max_depth=40):
        self.max_depth = max_depth
        self.root = Node()
        self.gain_function = gain_function

        indices = list(range(1, len(data[0])))

        self._split_recurs(self.root, data, indices)

        # Prune the tree recursively if there is a validation dataset
        if validation_data is not None:
            self._prune_recurs(self.root, validation_data)

    def predict(self, features):
        return self._predict_recurs(self.root, features)

    def accuracy(self, data):
        return 1 - self.loss(data)

    def loss(self, data):
        cnt = 0.0
        test_Y = [row[0] for row in data]
        for i in range(len(data)):
            prediction = self.predict(data[i])
            if (prediction != test_Y[i]):
                cnt += 1.0
        return cnt/len(data)

    def _predict_recurs(self, node, row):
        '''
        Predict the label given a row of features
        Traverse the tree until leaves to get the label
        '''
        if node.isleaf or node.index_split_on == 0:
            return node.label
        split_index = node.index_split_on
        if not row[split_index]:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)


    def _prune_recurs(self, node, validation_data):
        '''
        Prune the tree bottom up recursively
        '''
        if node.isleaf==False:
            if node.left is not None:
                self._prune_recurs(node.left, validation_data)
            if node.right is not None:
                self._prune_recurs(node.right, validation_data)
            if (node.left.isleaf) and (node.right.isleaf):
                # Prune node if loss is reduced
                old_loss = self.loss(validation_data)
                node.isleaf = True 
                new_loss = self.loss(validation_data)

                if new_loss > old_loss:
                    node.isleaf = False
                    
        
    def _is_terminal(self, node, data, indices):
        '''
        determine whether the node should stop splitting.
        Stop the recursion if:
            1. The dataset is empty.
            2. There are no more indices to split on.
            3. All the instances in this dataset belong to the same class
            4. The depth of the node reaches the maximum depth.
        '''
        y = [row[0] for row in data]
        if len(y)==0: 
            # no more indices to split on: return random label
            isleaf = True
            label = random.randint(0,1)
        else:
            condition = (len(indices)==0) or \
            (np.mean(y)==0) or (np.mean(y)==1) or \
            (node.depth == self.max_depth)
            
            if condition == True:
                isleaf = True
            else:
                isleaf = False
            
            label = (sum(y)/len(y)) >0.5

        return isleaf, label


    def _split_recurs(self, node, data, indices):
        '''
        Recursively split the node based on the rows and indices given
        '''
        # use _is_terminal() to check if the node needs to be split
        node.isleaf, node.label = self._is_terminal(node, data, indices) 

        if not node.isleaf:
            # select the column that has the maximum infomation gain to split on
            gain = np.zeros(len(indices))
            for i in range(len(indices)):
                gain[i] = self._calc_gain(data, indices[i], self.gain_function)

            node.index_split_on = indices[np.argmax(gain)]
            indices.remove(node.index_split_on)
        
            # Split the data and pass it recursively to the children
            left_data, right_data = [], []
            for row in data:
                if not row[node.index_split_on]:
                    left_data.append(row)
                else:
                    right_data.append(row)

            node.left = Node(depth=node.depth+1)
            node.right = Node(depth=node.depth+1)

            self._split_recurs(node.left, left_data, copy.copy(indices))
            self._split_recurs(node.right, right_data, copy.copy(indices))
        

    def _calc_gain(self, data, split_index, gain_function):
        '''
        Calculate the gain of the proposed splitting
        Gain = C(P[y=1]) - P[x_i=True] * C(P[y=1|x_i=True]) - P[x_i=False] * C(P[y=0|x_i=False])
        Where C(p) is the gain_function
        '''
        y = [row[0] for row in data]
        xi = [row[split_index] for row in data]

        if len(y) != 0 and len(xi) != 0:
            pxi_T = sum(xi)/len(xi)
            if pxi_T != 0 and pxi_T != 1:
                gain = gain_function(sum(y)/len(y)) - \
                pxi_T * gain_function(sum((xi[j]==True) and (y[j]==1) for j in range(len(y)))/len(y) / pxi_T) - \
                (1-pxi_T) * gain_function(sum((xi[j]==False) and (y[j]==0) for j in range(len(y)))/len(y) / (1-pxi_T))
            else:
                gain = 0
        else:
            gain = 0
        return gain
