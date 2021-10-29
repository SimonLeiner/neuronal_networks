"""
Name: main.py in Project: pneuronal_network
Author: Simon Leiner
Date: 18.10.21
Description: executable file in the project
"""

import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import activations
from layers import Layer
import losses
import network
import optimazation

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def one_hot_encoding(values):
    """
    :param values: numpy array
    :return: one hot encoded numpy matrix
    """
    return np.squeeze(np.eye(10)[values.reshape(-1)])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def execute_function():
    """

    This function executes the script

    :return: None

    """

    print("--Starting the process--")
    print("-" * 10)

    # loading the minst Dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # make sure the data we are recieving are numpy arrays
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    # easy plotting for a nice overview
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
    plt.show()

    # some statistics of the dataset
    image_size = 28
    number_differnet_labels = 10
    image_pixels = 28 * 28

    # flatten the array
    train_X.shape = (train_X.shape[0], image_pixels)
    test_X.shape = (test_X.shape[0], image_pixels)

    # shape of dataset
    print("Shapes of the dataset:")
    # note (x,y,z) : x = number elements, y = rows of element, z = cols of element
    print(f"X_train:{str(train_X.shape)}")
    print(f"Y_train:{str(train_y.shape)}")
    print(f"X_test:{str(test_X.shape)}")
    print(f"Y_test:{str(test_y.shape)}")
    print("-" * 10)

    # might want to scale or normalize the data

    # one hot encode the target column
    train_y_one_hot = one_hot_encoding(train_y)
    test_y_one_hot = one_hot_encoding(test_y)

    # define the loss function
    loss = losses.Loss_MSE()

    # define the activation functions
    relu = activations.Activation_ReLU()
    softmax = activations.Activation_Softmax()

    # define the optimizer
    sgd = optimazation.SGD(learning_rate=0.001)

    # creating the network
    net = network.Network(loss, sgd)

    # define the input Layer
    input_layer = Layer(image_pixels, 400, "input")

    # activation function for the input layer
    input_activation_layer = activations.Activation_Layer(relu)

    # define the 4 hidden Layers and the activation function for each Layer:

    # 5 neurons
    layer_1 = Layer(400, 200, "hidden_L1")
    l_1_activation = activations.Activation_Layer(relu)

    # 3 neurons
    layer_2 = Layer(200, 100, "hidden_L2")
    l_2_activation = activations.Activation_Layer(relu)

    # 4 neurons
    layer_3 = Layer(100, 50, "hidden_L3")
    l_3_activation = activations.Activation_Layer(relu)

    # 5 neurons
    layer_4 = Layer(50, 25, "hidden_L4")
    l_4_activation = activations.Activation_Layer(relu)

    # output Layer with 10 differnet neurons: each for each class
    output_layer = Layer(25, number_differnet_labels, "output")
    output_activation_layer = activations.Activation_Layer(softmax)

    # all the layers ordered in a List
    layers = [input_layer, input_activation_layer, layer_1, l_1_activation, layer_2, l_2_activation, layer_3,
              l_3_activation
        , layer_4, l_4_activation, output_layer, output_activation_layer]

    # add the layers to out network
    for layer in layers:
        net.add_layer(layer)

    # train the model
    net.train(x_train=train_X, y_train=train_y_one_hot, number_batches=1, subbatch_size=int(len(train_X / (1 * 10))),
              number_epochs=3)

    # evaluation

    print("Training Error:")
    # run the train set through the network and get predictions
    predictions_train = net.run(train_X, train_y, train_y_one_hot)

    print("Test Error:")
    # run the test set througth the network and get the predictions
    predictions_test = net.run(test_X, test_y, test_y_one_hot)

    return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    execute_function()
