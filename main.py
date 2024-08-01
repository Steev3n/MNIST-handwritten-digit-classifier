import numpy as np

# This NN abstracts away the concept of the neuron and is basically just a bunch of matrix multiplication. It's basically what a NN is, really.
# I try to keep the indexing of the layers and its parameters consistent, with the first layer be index 0 and last be (nbLayers) - 1.

# Learning rate should be tweaked to produce best learning results
LEARNING_RATE = 0.1

# Initializes the parameters of the NN given the dimensions of the layers as
# an array.
def init_paramaters(layer_dims):
    model = {}

    for layer in range(1, len(layer_dims)):
        # Each row is the number of neurons in next layer, each column is number of neurons in previous layer
        model['W' + str(layer)] = np.random.rand(layer_dims[layer], layer_dims[layer - 1])
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
    return np.maximum(0, z)


# Returns softmaxed vector, using numerically stable softmax
def softmax(array):
    expArray = np.exp(array - np.max(array))

    return expArray / expArray.sum(axis=0)


# Forward propagation, x = pixel vals of 1 example, parameters = model.
def forward_propagation(x, model):
    # You have (L - 1) weight matrices, (L - 1) bias vectors, (L) Z vectors, (L - 1) Activation value vectors, for a total of 4L - 3. So L = (len(model) + 3) / 4
    nbLayers = (len(model) + 3) // 4

    model['Z0'] = x

    for layer in range(1, nbLayers - 1):
        model['Z' + str(layer)] = model['W' + str(layer)].dot(model['Z' + str(layer - 1)]) + (model['b' + str(layer)])
        model['A' + str(layer)] = relu(model['Z' + str(layer)])

    # Doing last layer separately for softmax
    model['Z' + str(nbLayers - 1)] = model['W' + str(nbLayers - 1)].dot(model['Z' + str(nbLayers - 2)]) + (model['b' + str(nbLayers - 1)])
    model['A' + str(nbLayers - 1)] = softmax(model['Z' + str(nbLayers - 1)])

    return model


# y = ground truth. Will use cross entropy loss. Recall ith CEL is -y_i * log(prediction_i)
def loss(y, model):
    nbLayers = (len(model) + 3) // 4

    return -np.sum(y * np.log(model['A' + str(nbLayers - 1)]))


# Derivative of the relu is really just a 0, 1 piecewise function
def relu_derivative(array):
    return (array >= 0) * 1


# This is where the biggest part of NNs play a role; backpropagation.
# Will follow BP algorithm laid out in bpalg.png
def backpropagation(y, model):
    nbLayers = (len(model) + 3) // 4

    # Getting first delta.
    deltas = {nbLayers - 1: model['A' + str(nbLayers - 1)] - y}

    # Getting the rest of the deltas.
    for layer in range(nbLayers - 2, 0, -1):
        deltas[layer] = np.dot(np.transpose(model['W' + str(layer + 1)]), deltas[layer + 1]) * relu_derivative(model['Z' + str(layer)])

    # Here we update the parameters of our model. Since first layer does not have an activation function, I will just use the raw input data.

    model['W1'] -= LEARNING_RATE * np.outer(deltas[1], model['Z0'])
    model['b1'] -= LEARNING_RATE * deltas[1]

    for layer in range(2, nbLayers):
        model['W' + str(layer)] -= LEARNING_RATE * np.outer(deltas[layer], model['A' + str(layer - 1)])
        model['b' + str(layer)] -= LEARNING_RATE * deltas[layer]


model = init_paramaters([3, 2, 3])
label = np.array([0,0,1])

for i in range(100):
    model = forward_propagation([1, 2, 3], model)
    lossval = loss(label, model)
    backpropagation(label, model)

    if i % 10 == 0:
        print("Loss:", lossval)