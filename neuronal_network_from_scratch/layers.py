"""
Name : layers.py in Project: Neuronal_Networks
Author : Simon Leiner
Date    : 22.07.2021
Description: Layer class
"""

import numpy as np

class Layer:

    """Class for creating Layers"""

    def __init__(self, n_neurons_in, n_neurons_out,name):

        """
        :param n_neurons_in: number of neurons in Layer : 784 at the beginning
        :param n_neurons_out: number of neurons of the next layer
        """

        # initialize with random weights : dim(number of neurons in Layer,number of neurons in next Layer)  with a gaussian distribution
        self.weights = np.random.randn(n_neurons_in, n_neurons_out)

        # initialize with 0 bias : dim(1,number of neurons in next Layer) : Note 1 bias for each neuron
        self.biases = np.zeros((1, n_neurons_out))

        # define a Layer name
        self.name = name

    def forward(self, input):

        """
        This function moves one step foreward: computes the output for a given input

        :param input: Input data

        Forward Propagation:
        neuron Layer ahead = sum (all input Layers * all input weights) + bias for the neuron layer ahead
        """

        # save the inputs
        # dim(1, number of neurons in the input Layer : 784)
        self.input = input

        # for Transposing the output in the backward method, we need to ensure a multidimensional array
        # eg (784,) -> (784,1)

        if self.input.ndim == 1:
            self.input = self.input.reshape((1,-1))

        # calculate the output: weights * x + b with inputs := x and biases := b

        # dot product: of 2 vectors x, y := x.T * y
        output = np.dot(input,self.weights)

        # add the biases
        self.output = output + self.biases

        # Note: output is an np.array with dim(1,number of neurons in the next layer
        # returns a vector: dim(1, number of neurons in the next Layer)
        return self.output

    def backward(self,error_gradient_output,optimizing_function):

        """
        This function moves one step backward: computes the derivative for each training sample

        Note:
        we got the derivative of the cost function in respect to the output Layer and want the derivatives
        in respect to the weights, biases and the previous layer

        Layer function: aL = wL * aL-1 + bL

        Dim example for just 2 Layers in the netire Network: Input and Output layer of the Minst Dataset

        The first Layer has  784 neurons : 1 datapoint is a vector of size(1,784)
        Last Layer has 10 neurons : (del. C / del. aL) in respect to the last Layer is a vector of size: (1,10)
        """

        # gradient in repsect to the weights
        # (delta cost / delta aL) * aL-1
        # dim(1,10) * dim(1,784) -> matrix of dim(784,10) -> dot product
        error_gradient_weights = np.dot(self.input.T,error_gradient_output)

        # gradient in repsect to the biases
        # (delta cost / delta aL) * 1 -> vector of dim(1,10)
        error_gradient_bias = error_gradient_output

        # gradient in repsect to the previous layer
        # (delta cost / delta aL) * wL
        # dim(1,10) * dim(784,10) -> matrix of dim(1,784)
        error_gradient_input = np.dot(error_gradient_output,self.weights.T)

        # apply the stochastic gradient decent and update the parameters

        # update the weights
        self.weights = optimizing_function(self.weights,error_gradient_weights)

        # update the bias
        # delta cost / delta bL <=> (delta cost / delta aL) * (delta aL / delta bL)
        # (delta cost / delta aL) = error_gradient_output
        # (delta aL / delta bL) = 1 : aL = wL * aL-1 + bL
        self.biases = optimizing_function(self.biases,error_gradient_bias)

        return error_gradient_input
