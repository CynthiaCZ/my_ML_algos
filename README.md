<!-- TABLE OF CONTENTS -->
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the Project</a>
      <ul>
        <li><a href="#tools-and-versions">Tools and Versions</a></li>
      </ul>
    </li>
    <li>
      <a href="#sections">Sections</a>
      <ul>
        <li><a href="#linear-regression">Linear Regression</a></li>
        <li><a href="#logistic-regression">Logistic Regression</a></li>
        <li><a href="#validation-and-regularization">Validation and Regularization</a></li>
        <li><a href="#decision-trees">Decision Treess</a></li>
        <li><a href="#naive-bayes">Naive Bayes</a></li>
        <li><a href="#support-vector-machine">Support Vector Machine</a></li>
        <li><a href="#kmeans-clustering">Kmeans Clustering</a></li>
      </ul>
    </li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>



<!-- ABOUT THE PROJECT -->
## About The Project
This GitHub repository hosts a collection of machine learning algorithms implemented from scratch using standard Python and NumPy. The code and data files are adapted from the author's implementations of the homework assignments from CSCI1420 Machine Learning and DATA2060 Machine Learning: from Theory to Algorithms at Brown University. Having completed the first course and served as a Teaching Assistant for the second course for a year, the author brings a depth of understanding and practical experience to these implementations. The project encompasses a diverse set of models, including Linear Regression, Logistic Regression, Validation and Regularization, Decision Trees, Naive Bayes, Support Vector Machine, and Kmeans Clustering. Each algorithm is accompanied by a separate section in the readme file as detailed below, providing information on the datasets used, the functions within the model files, and the main programs for execution. The models are designed to address various classification and regression tasks, ranging from predicting wine quality ratings to classifying email spam.

### How to Run
* To run each machine learning algorithm, go to the corresponding folder and run the main.py file. This will call the supporting functions in the models.py files as well as other supporting scripts.
* The main scripts will also apply the models to datasets and report the model's performance in stdout.
* Some of the algorithms also have plots and diagrams as outputs. Those will be saved in the plot sub-folder within each algorithm folder. Sample outputs are included in the plot folders.


### Tools and Versions

The project uses the following tools and packages:
- python/3.10.7
- numpy/1.23.3
- matplotlib/3.6.0
- pandas/1.4.2
- quadprog/0.1.11
- scikit-learn/1.1.1

<!-- SECTIONS -->
## Sections
### Linear Regression
* data: wine quality dataset https://archive.ics.uci.edu/ml/datasets/Wine+Quality.
* models.py: contains functions to train, predict, and calculate MSE for the linear regression model.
* main.py: contains the main program to read the wine quality dataset, run the model, and print results. A wine quality rating (out of 10) is being predicted based on 11 attributes.
* output.png: sample output of training and testing MSE. 

### Logistic Regression
* data: UCI Census Income data from 1994 https://archive.ics.uci.edu/ml/datasets/Census+Income
* models.py: contains functions to train, predict, and calculate accuracy for the logistic regression model.
* main.py: contains the main program to read the census dataset, run the model, and print results. Education levels (3 levels) of individuals are being predicted based on the attributes from the census.
* output.png: sample output of epoch loss, number of epochs, and test accuracy

### Validation and Regularization
* data: UCI Breast Cancer Wisconsin (Diagnostic) Data https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
* models.py: contains functions to evaluate models by splitting data into train/test/validation sets, and returns lists of training and validation errors with respect to a selection of lambda hyperparameter values. Also contains functions to evaluate models by implementing k-fold cross validation, and returns a list of errors with respect to each value of lambda.
* main.py: contains the main program to read the breast cancer dataset, run the models, and print results. Whether or not a given patient has breast cancer is predicted from their health data.
* output.png: sample output of train and validation accuracy
* output_plot.png: k-fold validation result: train, validation, and k-fold errors at different lambda values.

### Decision Trees
* data: Spambase Dataset https://archive.ics.uci.edu/dataset/94/spambase and Chess Dataset https://archive.ics.uci.edu/dataset/21/chess+king+rook+vs+king+knight
* models.py: contains functions to implement decision trees for binary classification problems, including recursively splitting and pruning the trees.
* main.py: contains the main program to read data, run the model, and print results.
   The model is being applied to:
   1. Spambase Dataset: the goal is to train a model that can classify whether an email is spam or not from feature attributes such as the frequency of certain words and the amount of capital letters in a given message.
   2. Chess Dataset
    Each row of the chess.csv dataset contains 36 features, which represent the current state of the chess board. The task is to use the Decision Trees to classify whether or not it is possible for white to win the game.
* output.png: sample output of training and test loss of pruned and unpruned trees with different node scores.

### Naive Bayes
* data: German Credit dataset https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
* models.py: contains functions to implement a Naive Bayes model as well as different methods to gauge the trained model's fairness.
* main.py: contains the main program to read the German credit dataset, run classifiers, and print results. The prediction task is to predict whether someone's credit is good (1) or bad (0) based on attributes like sex, age, and personal status.
* output.png: train accuracy, test accuracy, and different fairness measures.

### Support Vector Machine
* data: Spambase dataset https://archive.ics.uci.edu/dataset/94/spambase
* models.py: contains functions to implement support vector machine (SVM) model to solve binary classification problems. A Python library for solving quadratic problems, quadprog, is used as an optimizer instead of gradient-based methods.
* qp.py: this file contains functions to solve quadratic quadratic problems using quadprog.
* main.py: contains the main program to read the spambase dataset, run the model, and print results. The goal is to train a model that can classify whether an email is spam or not from feature attributes such as the frequency of certain words and the amount of capital letters in a given message. 
* output.png: train and test accuracy of linear, RBF, and polynomial kernel.
* plot: accuracy at different hyperparameter and plots of different kernels applied to toy datasets.

### Kmeans Clustering
* data: contains the digits.csv file, where each row is an observation of a hand-written digit in 0-9, containing a label in the first column, and 8*8=64 pixel values in the rest of the columns.
* models.py: contains functions to train and predict using a Kmeans classifier.
* kmeans.py: contains helper functions required by the K-means method via iterative improvement.
* main.py: contains the main program to read data, run the Kmeans classifier, and print results. The goal is to cluster the hand-written digits.
* output.png: shapes of training and testing data as well as model accuracy.
* plot_clusters.png: plotted cluster centers to resemble the 10 digits.

<!-- SUMMARY -->
## Acknowledgments
The code and data included in this repository are adapted from the author's implementations of the homework assignments of CSCI1420 Machine Learning and DATA2060 Machine Learning: from Theory to Algorithms at Brown University. It is important to note that this code is intended solely for educational purposes. All rights and ownership of the original materials belong to Brown University.
