"""
Name : losses.py in Project: Neuronal_Networks
Author : Simon Leiner
Date    : 22.07.2021
Description: Loss function classes
"""

import numpy as np

class Loss_MSE:

    """Class defining the Loss function mean squared Error"""

    def f_x(self, y_pred, y_true):

        """
        This function defines the MSE function:

        :param y_pred: output from the last Layer
        :param y_true: true values
        :return: Loss between the predicted and true Labels

        Note: C = 1/n * sum(y_pred - y_true)^2
        """

        # scalar
        return np.mean(np.power(y_true - y_pred, 2))

    def f_prime_x(self, y_pred, y_true):

        """
        This function defines the derivative of the MSE function in respect to the output predictions:

        :param y_pred: output from the last Layer
        :param y_true: true values
        :return: Loss between the predicted and true Labels

        Note:
        derivative by the output predictions, order changes through (*-1) by chain rule
        C = 1/n * [(y1_true - y1_pred)^2 + ... + (yn_true - yn_pred)^2]
        delta C / delta y1_pred = 1/n * 2 * (y1_true - y1_pred) * -1  for all y 1,...,n
        """

        # partial deriviative for all y_true : vector
        return (2 / len(y_true)) * (y_pred - y_true)

class Loss_CategoricalCrossentropy():

    """
    Class defining the cost function cross entropy

    See: http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html
    """

    def f_x(self, y_pred, y_true):

        """
        This function defines the CC function:

        :param y_pred: output from the last Layer
        :param y_true: true values
        :return: Loss between the predicted and true Labels

        Note: Formula: L = y1 * log(y1_pred) + y1 * log(y1_pred) + ... + yk * log(yk_pred)
        """

        # if class targets: y_true is an numpy matrix , already one hot encoded => for each class a column

        # sum up predictions * true, over all rows: matrix multiplication, only 1 entry per row !=  => sum up rowise to get the value
        # y_pred: matrix with probabilities, y_true: matrix one hot encoded: just one 1 per row, 1 row for each sample
        # -> only the value where y_true = 1 "survives" and all other values are 0
        correct_confidences = np.sum(y_pred * y_true, axis=1)

        # calculate the negative log from the y_pred_clipped probabilities, index at the true class
        negative_log_likelihoods = -np.log(correct_confidences)

        # average loss as the mean of the numpy array with the losses
        data_loss = np.sum(negative_log_likelihoods)

        # scalar
        return data_loss

    def f_prime_x(self, y_pred, y_true):

        """
        This function defines the derivative of the CC function:

        :param y_pred: output from the last Layer
        :param y_true: true values
        :return: Loss between the predicted and true Labels

        Note:
        Formula: L = y1_true * log(y1_pred) + y2_true * log(y2_pred) + ... + yk_true * log(yk_pred)
        We recieve k partial derivatives in respect to the predicted output
        for each partial derivative only 1 summand "survives":
        delta L / delta y1_pred = y1_true * 1 / y1_pred
        """
        
        # if class targets: y_true is an numpy matrix , already one hot encoded => for each class a column

        # sum up predictions * true, over all rows: matrix multiplication, only 1 entry per row !=  => sum up rowise to get the value
        # y_pred: matrix with probabilities, y_true: matrix one hot encoded: just one 1 per row, 1 row for each sample
        # -> only the value where y_true = 1 "survives" and all other values are 0
        correct_confidences = np.sum(y_pred * y_true, axis=1)

        # array containing ones for 1/x
        ones = np.ones(shape=correct_confidences.shape)
        
        # divide the 2 arrays: 1 / y_pred
        derivative = np.divide(ones,correct_confidences)

        # vector with partial derivatives
        return derivative
