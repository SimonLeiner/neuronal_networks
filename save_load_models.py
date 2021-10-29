"""
Name : save_load_models.py in Project: Neuronal_Networks
Author : Simon Leiner
Date    : 17.07.2021
Description: functions for saving and loading models : https://keras.io/api/models/model_saving_apis/
"""

import os.path
from tensorflow.keras.models import load_model

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def save_model_to_path(trained_model,path):

    """

    This function saves the architecture of the model, the weights, configuration, state of the optimizer.

    :param trained_model: object : finished trained model to save
    :param path: string : path to save the model
    :return: None

    """

    # if file isn't already existing
    if os.path.isfile(path) is False:

        # save the model into the new path
        trained_model.save(path)

    return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def load_model_from_path(path):

    """

    This function loads an already configured path saved at a given path.

    :param path: string : path where the model is saved
    :return: object : model returns the model

    """

    # get the model
    model = load_model(path)

    # print the summary of the model to show everything is correct:
    print(model.summary())
    print("-" * 10)

    return model

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #