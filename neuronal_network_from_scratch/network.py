"""
Name : network.py in Project: Neuronal_Networks
Author : Simon Leiner
Date    : 22.07.2021
Description: Network class
"""

import numpy as np

# only need pandas for printing
import pandas as pd

class Network:

    def __init__(self,loss_class,optim_class):

        """
        :param loss_class: Object witch contains the loss function
        :param optim_class: Object witch contains the optimizing function
        """
        self.layers = []
        self.loss = loss_class.f_x
        self.loss_prime = loss_class.f_prime_x
        self.optimize = optim_class.optimize

    def add_layer(self,layer):

        """
        :param layer: A Layer Object
        """

        self.layers.append(layer)

    def run(self,input,train_y,train_y_one_hot):

        """
        This function defines the derivative of the CC function:

        :param input: input Data
        :param train_y: true values for camparison
        :param train_y_one_hot: true values encoded for loss calculation
        """

        # empty array for saving values
        results = []

        # list for saving the losses
        error_list = []

        # Note: we run each datapoint through the network and calculate the loss for that Datapoint
        # for each sample do: should be 60000 or 10000 here
        for i in range(len(input)):

            # move through the layers
            for layer in self.layers:

                # for the input layer
                if layer.name == "input":

                    # define the input for the input Layer as the output from an abstract earlier Layer
                    output = input[i]

                # move one step foreward always from the output of the previous Layer
                output = layer.forward(output)

            # calculate the loss for each data point
            loss_ = self.loss(output, train_y_one_hot[i])

            # append each error
            error_list.append(loss_)

            # always appends the last resulting output for each input data
            results.append(output)

        # convert the results to a numpy array:
        results = np.array(results)

        # flatten the array
        results.shape = (len(input),10)

        self.result = results

        # printing the results
        print(f"Predictions of the first 9 Datapoints:")
        df = pd.DataFrame(data= self.result)
        df["True value"] = train_y
        df = df.round(4)
        print(df.head(9))
        print("-" * 10)

        # calcualte the average forward error for printing reasons:
        average_error = sum(error_list) / len(error_list)

        print(f"The average Loss Mean Squared Error from the NN is: {round(average_error,4)*100} % ")
        print("-" * 10)

        return self.result

    def train(self,x_train, y_train,number_batches,subbatch_size,number_epochs):

        """
        This function runs forward through the network and collects the outcoming resultst: predictions

        :param input: input Data

        # Note: https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
        """

        # combine the X and y data
        # See: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
        # See: https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html
        # data_full = np.concatenate((x_train,y_train), axis = 1)
        data_full = np.column_stack((x_train, y_train))
        # print(data_full)

        # shuffle the data first : shuffles inplace
        # See: https://numpy.org/doc/1.20/reference/random/generated/numpy.random.shuffle.html
        np.random.shuffle(data_full)
        # print(data_full)

        # split the data into a given number of batches
        # See: https://numpy.org/doc/stable/reference/generated/numpy.split.html#numpy.split
        batches = np.split(data_full,number_batches)
        # print(batches)

        print("Starting training the model")
        print("-" * 10)

        # run the train method number_epochs times:
        for j in range(number_epochs):

            # list for saving the losses
            error_list = []

            # loop over batches
            # Note: these batches are just divided, not randomly selected
            for batch in batches:

                # draw a random subsample of size x
                # See: https://www.kite.com/python/answers/how-to-select-random-rows-from-a-numpy-array-in-python
                random_subsample = batch[np.random.choice(batch.shape[0], size = subbatch_size, replace = False)]
                # print(random_subsample.shape)

                # split the data again : split at the 784 column
                # See: https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html
                x_train_batch, y_train_batch = np.hsplit(random_subsample, [int(x_train.shape[1])])
                # print(x_train_batch.shape)
                # print(y_train_batch.shape)

                # Note: we run each datapoint through the network and calculate the loss for that Datapoint
                # for each sample do: should be 60000 or 10000 here
                for i in range(len(x_train_batch)):

                    # move through the layers
                    for layer in self.layers:

                        # for the input layer
                        if layer.name == "input":
                            # define the input for the input Layer as the output from an acstract earlier Layer
                            output = x_train_batch[i]

                        # move one step foreward always from the output of the previous Layer
                        output = layer.forward(output)

                    # calculate the loss for each data point
                    loss_ = self.loss(output, y_train_batch[i])

                    # append each error
                    error_list.append(loss_)

                    # calcualte the derivative of the error with respect to the last output layer: the predictions
                    # delta cost / delta aL
                    error = self.loss_prime(output, y_train_batch[i])

                    # for each layer in reversed order: starting at the output layer
                    for layer in reversed(self.layers):

                        # move one step backwards
                        error = layer.backward(error,self.optimize)

            # calcualte the average forward error for printing reasons:
            average_error = sum(error_list) / len(error_list)

            print(f"The average Error of all training datapoints for Epoch {j} is: {average_error}")
            print("-" * 10)

        print("Finished training the model")
        print("-" * 10)

        return None

