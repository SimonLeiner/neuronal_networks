"""
Name : optimazation.py in Project: Neuronal_Networks
Author : Simon Leiner
Date    : 23.07.2021
Description: Optimization file with computes the Stochastic Gradient Decent Algorithm
"""

class SGD:

    """Class for creating Layers"""

    def __init__(self,learning_rate):

        """
        :param learning_rate: constant learning rate for the Algorithm
        """

        self.learning_rate = learning_rate

    def optimize(self,input,gradient):

        """
        This function computes the Gradient Decent Algorithm for one step:
        :param input: one of layer, weights, biases
        :param input: associated gradient

        Note:
        x(k+1) = x(k) + learning_rate * - gradient(x(k))

        """

        return input - self.learning_rate * gradient
