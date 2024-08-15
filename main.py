import math
import random

import numpy as np
import matplotlib.pyplot as plt
# MNIST dataset loaded from the keras package.
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras

# This NN abstracts away the concept of the neuron and is basically just a bunch of matrix multiplication. It's basically what a NN is, really.
# I try to keep the indexing of the layers and its parameters consistent, with the first layer be index 0 and last be (nbLayers) - 1.

# Hyperparameters
LEARNING_RATE = 0.000133
WEIGHT_SCALING = 0.8
BATCH_SIZE = 60
NB_EPOCHS = 50
TRAINING_SET_COUNT = 60000
PARTIAL_NORMALIZATION = 1

TESTING_SET_COUNT = 10000

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# Initializes the parameters of the NN given the dimensions of the layers as
# an array.
def init_paramaters(layer_dims):
    model = {}

    for layer in range(1, len(layer_dims)):
        # Each row is the number of neurons in next layer, each column is number of neurons in previous layer
        model['W' + str(layer)] = np.random.normal(0, math.sqrt(2 / layer_dims[layer - 1]), (layer_dims[layer], layer_dims[layer - 1])) * WEIGHT_SCALING
        # Length = number of neurons in next layer
        model['b' + str(layer)] = np.zeros(layer_dims[layer])
        # NOTE: First layer does not have an activation value. Hence the reason I initialize it here.
        model['A' + str(layer)] = np.zeros(layer_dims[layer])

    for layer in range(len(layer_dims)):
        # Adds Z and activation vectors to our model.
        model['Z' + str(layer)] = np.zeros(layer_dims[layer])

    return model


# Returns vector of elementwise maximum between 0 and the vector component.
def relu(z):
    return np.where(z >= 0, z, 0.02 * z)


# Forward propagation, x = pixel vals of 1 example, parameters = model.
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


# y = ground truth. Will use MSE.
def loss(y, model):
    nbLayers = (len(model) + 3) // 4
    one_hot_encoding = np.zeros(len(model['A' + str(nbLayers - 1)]))
    one_hot_encoding[y] = 1
    prediction = model['A' + str(nbLayers - 1)]
    loss = 0

    for i in range(len(prediction)):
        loss += (one_hot_encoding[i] - prediction[i]) ** 2

    regLoss = loss / 2

    return regLoss


# Derivative of the relu is really just a 0, 1 piecewise function
def relu_derivative(array):
    return np.where(array >= 0, 1, 0.02)


# This is where the biggest part of NNs play a role; backpropagation.
# Will follow BP algorithm laid out in bpalg.png
def backpropagation(y, model):
    nbLayers = (len(model) + 3) // 4
    partialDerivatives = {}

    one_hot_encoding = np.zeros(len(model['Z' + str(nbLayers - 1)]))
    one_hot_encoding[y] = 1

    # Getting first delta.
    deltas = {nbLayers - 1: model['A' + str(nbLayers - 1)] - one_hot_encoding}

    # Getting the rest of the deltas.
    for layer in range(nbLayers - 2, 0, -1):
        deltas[layer] = np.dot(np.transpose(model['W' + str(layer + 1)]), deltas[layer + 1]) * relu_derivative(model['Z' + str(layer)])

    # Here we get the first set of partial derivatives. Since first layer does not have an activation function, I will just use the raw input data.
    partialDerivatives['W1'] = np.outer(deltas[1], model['Z0'])
    partialDerivatives['b1'] = deltas[1]

    for layer in range(2, nbLayers):
        partialDerivatives['W' + str(layer)] = np.outer(deltas[layer], model['A' + str(layer - 1)])
        partialDerivatives['b' + str(layer)] = deltas[layer]

    return partialDerivatives


def minibatchTrain():
    nbMiniBatches = TRAINING_SET_COUNT // BATCH_SIZE
    batchVector = []

    for i in range(nbMiniBatches):
        singleBatch = []
        for j in range(BATCH_SIZE):
            singleExample = []

            index = random.randint(0, TRAINING_SET_COUNT - 1)

            singleExample.append(x_train[index].flatten() / 255)
            singleExample.append(y_train[index])

            singleBatch.append(singleExample)

        batchVector.append(singleBatch)

    return batchVector


def combinePartials(singlePartialDerivative, totalPartialDerivatives):
    for key in singlePartialDerivative:
        if key in totalPartialDerivatives:
            totalPartialDerivatives[key] += PARTIAL_NORMALIZATION * singlePartialDerivative[key]
        else:
            totalPartialDerivatives[key] = PARTIAL_NORMALIZATION * singlePartialDerivative[key]

    return totalPartialDerivatives


def updateParams(model, partials, lr):
    for key in partials:
        model[key] -= np.clip(lr * partials[key], -0.5, 0.5)

    return model


def getTestLoss(model):
    nbLayers = (len(model) + 3) // 4
    nbRightPrediction = 0

    for index in range(TESTING_SET_COUNT):
        modelAfter = forward_propagation(x_test[index].flatten(), model)

        maxVal = np.max(modelAfter['A' + str(nbLayers - 1)])
        maxIndex = np.where(modelAfter['A' + str(nbLayers - 1)] == maxVal)[0]

        if y_test[index] == maxIndex:
            nbRightPrediction += 1

    return nbRightPrediction / TESTING_SET_COUNT


def train(model):
    batchesVector = minibatchTrain()
    x_axis = []
    y_axis_loss = []
    y_axis_lr = []
    x_axis_test = []
    y_axis_test = []
    figure, axis = plt.subplots(2)

    # What we have now is essentially 2 tuples of tuples of training examples.
    nbMiniBatches = TRAINING_SET_COUNT // BATCH_SIZE
    tempModel = model
    totalNbBatchesTrained = 0

    for epoch in range(NB_EPOCHS):

        for i in range(nbMiniBatches):

            batchPartialDerivative = {}
            batchLoss = 0

            for j in range(BATCH_SIZE):
                tempModel = forward_propagation(batchesVector[i][j][0], tempModel)
                partialDerivativeOneExample = backpropagation(batchesVector[i][j][1], tempModel)
                batchPartialDerivative = combinePartials(partialDerivativeOneExample, batchPartialDerivative)

                batchLoss += loss(batchesVector[i][j][1], tempModel)

                #if j % 10000 == 0: print(f"Epoch: {epoch + 1}, Example: {j + 1} out of {BATCH_SIZE}")

            lr = LEARNING_RATE / (1.4 ** (epoch // 5))
            y_axis_lr.append(lr)

            tempModel = updateParams(tempModel, batchPartialDerivative, lr)
            totalNbBatchesTrained += 1

            x_axis.append(totalNbBatchesTrained)
            lossVal = batchLoss / BATCH_SIZE
            y_axis_loss.append(lossVal)

            print(f"Epoch: {epoch + 1} out of {NB_EPOCHS}, Minibatch: {i + 1} out of {nbMiniBatches}")

    axis[0].plot(x_axis, y_axis_loss)
    axis[0].set_title("Training loss")
    axis[1].plot(x_axis, y_axis_lr)
    axis[1].set_title("Learning rate")
    plt.show()

    return tempModel


def afterTrain(model):
    nbLayers = (len(model) + 3) // 4

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

    np.save('model.npy', model)

    plt.show()


def load_model():
    model = np.load('model (95%).npy', allow_pickle='TRUE').item()

    return model


model = init_paramaters([784, 385, 385, 10])

# model = train(model)
afterTrain(load_model())
print("Accuracy:", getTestLoss(load_model()))
