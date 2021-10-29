"""
Name : activations.py in Project: Neuronal_Networks
Author : Simon Leiner
Date    : 22.07.2021
Description: activation function classes
"""

import numpy as np

class Activation_ReLU:

    """Class defining function Relu"""

    def f_x(self, inputs):

        """This function defines the Relu activation function:
        :param inputs: input data
        :return: result of applied function
        """

        # use np.max on array
        return np.maximum(0, inputs)

    def f_prime_x(self,inputs):

        """This function defines the derivative of the Relu activation function:
        :param inputs: input data
        :return: result of applied function
        """

        # if x < 0:
        #     y = 0
        # elif x == 0:
        #     y = undefined
        # else:
        #     y = 1
        # use numpy.heaviside as it achieves the same

        # use np.heaviside on array
        # See: https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html
        return np.heaviside(inputs,0.5)

        # note if x == 0, the function is undefinied and we pass the assign the value 0.5 here

class Activation_Softmax:

    """
    Class defining function Softmax
    Note: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    Softmax is fundamentally a vector function. It takes a vector as input and produces a vector as output
    """

    def f_x(self, inputs):

        """
        This function defines the softmax activation (stable softmax) function:

        :param inputs: input data
        :return: probabilities  of the classes for each input

        """

        # exponentiate the values : no negative values anymore, but substract the biggest value before in order to set the highest possible Input as exp(0) = 1
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # normalize in order to get the probabilities that sum up to one
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        # vector
        return probabilities

    def f_prime_x(self,inputs):

        """This function defines the derivative of the softmax activation function:
        :param inputs: input data

        Note: can be derived for via the chainrule:
        See: https://stats.stackexchange.com/questions/370723/how-to-calculate-the-derivative-of-crossentropy-error-function
        """

        # get the function values: array
        function_values = self.f_x(inputs)

        # delta S = S - S^2
        derivative_function = lambda x: (x - x**2)

        # vectorize the formula
        derivative_function = np.vectorize(derivative_function)

        # apply the formula to the numpy array
        derivative = derivative_function(function_values)

        return derivative

class Activation_Layer:

    """Class for the of the Activation function: Treat the application of the activation function as a new layer """

    def __init__(self, activation_class):

        """
        :param activation_class: Class defining the activation and derivative of the Activation function
        """

        # get the both in the class defiened functions
        self.activation = activation_class.f_x
        self.activation_prime = activation_class.f_prime_x

        # define a Layer name
        self.name = "activation"

    def forward(self, input):

        """
        This function applies the activation function

        :param input: Input data

        Forward Propagation:
        neuron Layer value = actifation_function(neuron Layer value)
        """

        # save the inputs
        self.input = input

        # apply the activation function
        self.output = self.activation(input)

        # Note: output is an np.array with dim(1,number of neurons in the next layer
        return self.output

    def backward(self,error_gradient_output,optimizing_function):

        """
        This function applies the derivative of the activation function

        Note:
        we treat the activation as a new layer with a different Layer function: aL = f(aL-1)
        -> no derivative in respect to the weights and biases: no learning
        -> only derivative in respect to the previous Layer
        """

        # gradient in repsect to the weights
        # error_gradient_weights = 0

        # gradient in repsect to the biases
        # error_gradient_bias = 0

        # Note:
        # (delta cost / delta aL-1) = (delta cost / delta aL) * (delta aL / delta aL-1)
        # (delta cost / delta aL) = error_gradient_output
        # (delta aL / delta aL-1) = inverse of activation function
        # gradient in repsect to the previous layer
        error_gradient_input = error_gradient_output * self.activation_prime(self.input)

        return error_gradient_input

