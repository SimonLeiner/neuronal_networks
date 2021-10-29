"""
Name : tensorflow_example.py in Project: Neuronal_Networks
Author : Simon Leiner
Date    : 17.07.2021
Description: getting familiar with tensorflow and keras: https://keras.io/api/
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import categorical_crossentropy

# Example for Cat and Dog picture classification

# create the model by grouping a linear stack of layers into a tf.keras.Model : https://keras.io/api/models/sequential/
model = Sequential(layers=[...], name = "Modelname")

# the add method : Adds a layer instance on top of the layer stack
model.add("layer")

# pop method : like in queues : removes the last Layer from the model
model.pop("layer")

# check the model
print(model.summary())

# train the model : https://keras.io/api/models/model_training_apis/

# batch size : Number of samples per gradient update : very important for the optimizer
batch_size = 1

# epochs : An epoch is an iteration over the entire x and y data provided
epochs = 1

#usse SGC or adma, as well as differnet metrics and loss functions
model.compile(optimizer= SGD, loss = "categorical_crossentropy",metrics = "accuracy")

# fit the model with the training date
model.fit(x =..., y=...,batch_size = batch_size, epochs = epochs, validation_split=0.1)

# predict with the model
predictions = model.predict(x = ...,batch_size = 2,verbose = 0)

# now we can use our typical classification metrics to evaluate the model

# specified in the model.compile function : eg accuracy:
score = model.evaluate(...,...,verbose=0)

print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")



