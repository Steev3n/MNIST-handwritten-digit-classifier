import math
import random

import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
# MNIST dataset loaded from the keras package.
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras


def load_model():
    model = np.load('model.npy', allow_pickle='TRUE').item()

    return model


def relu(z):
    return np.where(z >= 0, z, 0.02 * z)


def forward_propagation(x, model):
    # You have (L - 1) weight matrices, (L - 1) bias vectors, (L) Z vectors, (L - 1) Activation value vectors, for a total of 4L - 3. So L = (len(model) + 3) / 4
    nbLayers = (len(model) + 3) // 4

    model['Z0'] = x

    model['Z1'] = model['W1'].dot(model['Z0']) + (model['b1'])
    model['A1'] = relu(model['Z1'])

    for layer in range(2, nbLayers - 1):
        model['Z' + str(layer)] = model['W' + str(layer)].dot(model['A' + str(layer - 1)]) + (model['b' + str(layer)])
        model['A' + str(layer)] = relu(model['Z' + str(layer)])

    # Doing last layer separately for sigmoid and normalization of last Z. -------------------------- TOOK OUT SOFTMAX
    intermediateZ = model['W' + str(nbLayers - 1)].dot(model['Z' + str(nbLayers - 2)]) + (model['b' + str(nbLayers - 1)])
    model['Z' + str(nbLayers - 1)] = intermediateZ
    model['A' + str(nbLayers - 1)] = (model['Z' + str(nbLayers - 1)])

    return model


def digitClassifier(model):
    nbLayers = (len(model) + 3) // 4
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    num_row = 2
    num_col = 5

    # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(10):
        index = random.randint(0, 10000)
        modelAfter = forward_propagation(x_test[index].flatten(), model)

        maxVal = np.max(modelAfter['A' + str(nbLayers - 1)])
        maxIndex = np.where(modelAfter['A' + str(nbLayers - 1)] == maxVal)[0]

        ax = axes[i // num_col, i % num_col]
        ax.imshow(x_test[index], cmap='gray')
        ax.set_title(f'Prediction: {maxIndex}')

    plt.savefig('10examples.png')


digitClassifier(load_model())
