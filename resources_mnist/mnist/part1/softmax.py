import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

#pragma: coderesponse template
def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    #YOUR CODE HERE
    # raise NotImplementedError
    print('Prob')
    n_rows = X.shape[0]
    n_labels = theta.shape[0]

    # theta_x = np.empty((n_labels, n_rows))
    # np.matmul(theta, (1.0/temp_parameter)*X.T, out=theta_x)
    theta_x = np.matmul(theta, X.T/temp_parameter)
    # theta_x = np.exp(np.matmul(theta, X.T/temp_parameter))

    # get the normalisation (the max across the cols of theta_x) and broadcast  to n by n
    # Cs = np.empty((n_rows))
    # np.max(theta_x, axis=0, out=Cs)
    Cs = np.max(theta_x, axis=0).reshape(1, -1) #.repeat(n_labels, axis=0)

    # exp_theta_x = np.empty((n_labels, n_rows))
    # np.exp(theta_x - Cs.reshape(1,-1), out=exp_theta_x)
    exp_theta_x = np.exp(theta_x - Cs)

    # H = exp_theta_x / np.sum(exp_theta_x, axis=0)
    # H = 
    # print('done with prob')
    return exp_theta_x / np.sum(exp_theta_x, axis=0)

#pragma: coderesponse end

#pragma: coderesponse template
def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    print('in cost')
    #YOUR CODE HERE
    # raise NotImplementedError
    n_rows = X.shape[0]
    # n_labels = theta.shape[0]
 
    log_H = np.log(np.clip(compute_probabilities(X, theta, temp_parameter), 1e-15, 1-1e-15))
    # M = sparse.coo_matrix(([1]*n_rows, (Y, range(n_rows))), shape=(n_labels, n_rows)).toarray()
    # M = (np.arange(n_labels).reshape(-1, 1) == Y)
    # cost = np.sum((M * log_H))/-n_rows +\
    #      (lambda_factor/2)*np.sum(theta**2)
    
    # faster
    # cost = 
    # cost = miss_class_err + reg_err
    print('done with cost')
    return - np.mean(log_H[Y, np.arange(n_rows)]) + (lambda_factor/2)*np.sum(theta**2)

#pragma: coderesponse end

#pragma: coderesponse template
def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    #YOUR CODE HERE
    print('in grad')
    #raise NotImplementedError
    # n_labels = theta.shape[0]
    n_rows = X.shape[0]
    # n_cols = X.shape[1]

    # sparse matrix of ones and zeros
    # each column has a one in a position corresponding to the label
    # M = sparse.coo_matrix(([1]*n_rows, (Y, range(n_rows))), shape=(n_labels, n_rows)).toarray()
    H = - compute_probabilities(X, theta, temp_parameter)

    # great?
    #theta_ = np.matmul((np.arange(n_labels).reshape(-1, 1) == Y) - H, X)
    H[Y, np.arange(n_rows)] = 1.0 + H[Y, np.arange(n_rows)]
    # theta_ = 
    # here is a very dense computation :)
    # theta = theta - alpha*((-1.0/(temp_parameter*n_rows))* \
    #     np.sum(X * (M - H).reshape(n_labels, n_rows, 1), axis=1) + \
    #         lambda_factor * theta)
    
    theta = theta - alpha*(np.matmul(H,X) / (-temp_parameter*n_rows) + lambda_factor * theta)
    print('done with grad')
    return theta

#pragma: coderesponse end

#pragma: coderesponse template
def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    #YOUR CODE HERE
    # raise NotImplementedError
    train_y_mod3 = train_y % 3
    test_y_mod3 = test_y % 3

    return train_y_mod3, test_y_mod3
#pragma: coderesponse end

#pragma: coderesponse template
def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    #YOUR CODE HERE
    # raise NotImplementedError
    # X = augment_feature_vector(X)
    Y_hat = get_classification(X, theta, temp_parameter)
    Y_hat_mod3 = Y_hat % 3
    error = 1 - np.mean(Y_hat_mod3 == Y)

    return error

#pragma: coderesponse end

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """

    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
