import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import glob
import sys

from helpers import *

# you shouldn't need to make any more imports

class NeuralNetwork(object):
    """
    Abstraction of neural network.
    Stores parameters, activations, cached values.
    Provides necessary functions for training and prediction.
    """
    def __init__(self, layer_dimensions, drop_prob=0.0, reg_lambda=0.0,
            weights=None, biases=None, activation=None):
        """
        Initializes the weights and biases for each layer
        :param layer_dimensions: (list) number of nodes in each layer
        :param drop_prob: drop probability for dropout layers. Only required in part 2 of the assignment
        :param reg_lambda: regularization parameter. Only required in part 2 of the assignment
        """
        seed = 1
        np.random.seed(seed)

        self.parameters = {}
        self.num_layers = len(layer_dimensions)
        self.drop_prob = drop_prob
        self.reg_lambda = reg_lambda
        self.activation = activation
        # init parameters

        if weights is not None:
            self.parameters['weights'] = weights
        else:
            self.parameters['weights'] = [np.random.rand(layer_dimensions[idx], val).T
                for idx, val in enumerate(layer_dimensions[1:])]

        print("weights[0] shape: %s" % str(self.parameters['weights'][0].shape))

        if biases is not None:
            self.parameters['biases'] = biases
        else:
            self.parameters['biases'] = [np.random.rand(n, 1)
                for n in layer_dimensions[1:]]

    def affineForward(self, A, W, b):
        """
        Forward pass for the affine layer.
        :param A: input matrix, shape (L, S), where L is the number of hidden units in the previous layer and S is
        the number of samples
        :returns: the affine product WA + b, along with the cache required for the backward pass
        """
        #or maybe join b into W

        print("Shapes: %s, %s" % (str(A.shape), str(W.shape)))
        return W * A  , np.zeros(3) # + b, np.zeros(3)

    def activationForward(self, A, activation="relu"):
        """
        Common interface to access all activation functions.
        :param A: input to the activation function
        :param prob: activation funciton to apply to A. Just "relu" for this assignment.
        :returns: activation(A)
        """
        if self.activation is not None:
            return self.activation(A)

        relu_v = np.vectorize(self.relu)
        return relu_v(A)

    def relu(self, X):
        return np.maximum(0,X)

    def dropout(self, A, prob):
        """
        :param A:
        :param prob: drop prob
        :returns: tuple (A, M)
            WHERE
            A is matrix after applying dropout
            M is dropout mask, used in the backward pass
        """

        return A, M

    def forwardPropagation(self, X):
        """
        Runs an input X through the neural network to compute activations
        for all layers. Returns the output computed at the last layer along
        with the cache required for backpropagation.
        :returns: (tuple) AL, cache
            WHERE
            AL is activation of last layer
            cache is cached values for each layer that
                     are needed in further steps
        """
        cache = np.zeros(3)

        layerout = np.asmatrix(X)
        for l in range(self.num_layers - 1):
            layerout, cache = self.affineForward(layerout,
                self.parameters['weights'][l], self.parameters['biases'][l])
        layerout = self.activationForward(layerout)

        return layerout, cache

    def costFunction(self, AL, y):
        """
        :param AL: Activation of last layer, shape (num_classes, S)
        :param y: labels, shape (S)
        :param alpha: regularization parameter
        :returns cost, dAL: A scalar denoting cost and the gradient of cost
        """
        # compute loss

        if self.reg_lambda > 0:
            # add regularization
            pass


        # gradient of cost
        dAL = 0
        return cost, dAL

    def affineBackward(self, dA_prev, cache):
        """
        Backward pass for the affine layer.
        :param dA_prev: gradient from the next layer.
        :param cache: cache returned in affineForward
        :returns dA: gradient on the input to this layer
                 dW: gradient on the weights
                 db: gradient on the bias
        """

        return dA, dW, db

    def activationBackward(self, dA, cache, activation="relu"):
        """
        Interface to call backward on activation functions.
        In this case, it's just relu.
        """
        pass


    def relu_derivative(self, dx, cached_x):

        return dx

    def dropout_backward(self, dA, cache):

        return dA

    def backPropagation(self, dAL, Y, cache):
        """
        Run backpropagation to compute gradients on all paramters in the model
        :param dAL: gradient on the last layer of the network. Returned by the cost function.
        :param Y: labels
        :param cache: cached values during forwardprop
        :returns gradients: dW and db for each weight/bias
        """
        gradients = {}

        for i in range(10):


            if self.drop_prob > 0:
                #call dropout_backward
                pass


        if self.reg_lambda > 0:
            # add gradients from L2 regularization to each dW
            pass

        return gradients


    def updateParameters(self, gradients, alpha):
        """
        :param gradients: gradients for each weight/bias
        :param alpha: step size for gradient descent
        """
        pass

    def train(self, X, y, iters=1000, alpha=0.0001, batch_size=100, print_every=100):
        """
        :param X: input samples, each column is a sample
        :param y: labels for input samples, y.shape[0] must equal X.shape[1]
        :param iters: number of training iterations
        :param alpha: step size for gradient descent
        :param batch_size: number of samples in a minibatch
        :param print_every: no. of iterations to print debug info after
        """

        for i in range(0, iters):
            # get minibatch

            # forward prop

            # compute loss

            # compute gradients

            # update weights and biases based on gradient

            if i % print_every == 0:
                # print cost, train and validation set accuracies
                pass

    def predict(self, X):
        """
        Make predictions for each sample
        """

        return self.forwardPropagation(X)


    def get_batch(self, X, y, batch_size):
        """
        Return minibatch of samples and labels

        :param X, y: samples and corresponding labels
        :parma batch_size: minibatch size
        :returns: (tuple) X_batch, y_batch
        """

        return X_batch, y_batch


# test forward prop

def test_forward_prop_affine_one_output():
    weights = np.matrix("1, 2, 3")
    biases = [np.zeros((1,1))]
    net = NeuralNetwork([3, 1], weights=weights, activation=lambda x: x, biases=biases)
    res, cache = net.forwardPropagation(np.matrix("0;1;2"))
    assert np.array_equal(res, np.matrix("8"))

test_forward_prop_affine_one_output()

def test_forward_prop_affine_three_output():
    weights = [np.matrix("1, 2, 3; 4, 5, 6; 7, 8, 9")]
    biases = [np.zeros((3,1))]
    net = NeuralNetwork([3, 3], weights=weights, activation=lambda x: x, biases=biases)
    res, cache = net.forwardPropagation(np.matrix("1;2;3"))
    assert np.array_equal(res, np.matrix("14; 32; 50"))

test_forward_prop_affine_three_output()

def test_forward_prop_activation_three_output():
    weights = [np.matrix("1, 2, 3; 4, 5, 6; 7, 8, 9")]
    biases = [np.zeros((3,1))]
    activation = lambda x : np.matrix("2, 0, 0; 0, 2, 0; 0, 0, 2") * x
    net = NeuralNetwork([3, 3], weights=weights, activation=activation, biases=biases)
    res, cache = net.forwardPropagation(np.matrix("1;2;3"))
    assert np.array_equal(res, np.matrix("28; 64; 100"))

test_forward_prop_activation_three_output()