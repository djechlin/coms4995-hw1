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
        self.relu_v = np.vectorize(self.relu)
        # init parameters

        self.parameters['weights'] = [np.random.rand(layer_dimensions[idx-1], val)
            for idx, val in enumerate(layer_dimensions[1:])]

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

        print(W.shape)
        print(A.shape)
        print(b.shape)
        return W.transpose()*A + b

    def activationForward(self, A, activation="relu"):
        """
        Common interface to access all activation functions.
        :param A: input to the activation function
        :param prob: activation funciton to apply to A. Just "relu" for this assignment.
        :returns: activation(A)
        """
        if self.activation is not None:
            return self.activation(A)

        print("A shape %s " % str(A.shape))
        return self.relu_v(A)

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

        cache = []
        A = X
        W = self.parameters['weights']
        b = self.parameters['biases']

        # range(0, 3) = (0, 1, 2)
        # one fewer computation than layers
        for l in range(0, self.num_layers - 1):
            Z = self.affineForward(A, W[l], b[l])
            print("Z: %s" % str(Z.shape))
            A = self.activationForward(Z)
            cache.append(A)

        #softmax
        A_exp = np.exp(A)
        AL = A_exp / np.sum(A_exp)

        return AL, cache

    def costFunction(self, AL, y):
        """
        :param AL: Activation of last layer, shape (num_classes, S)
        :param y: labels, shape (S)
        :param alpha: regularization parameter
        :returns cost, dAL: A scalar denoting cost and the gradient of cost
        """
        # compute loss

        # log loss
        y_i = 0
        cost = 0
        for sample in range(AL.shape[1]):
            #if label = node #, then yTrue = 1
            for i, node in enumerate(sample):
                yTrue = 1 if y[y_i] == i else 0
                cost += -yTrue*log(sample[i]) - (1 - yTrue)*log(1 - sample[i])
            y_i+=1
        if self.reg_lambda > 0:
            pass
            # add regularization
            # TODO

        cost /= AL.shape(1)

        # gradient of cost #just subtract 1 from the correct class
        for i in range(0, AL.shape[0]):
            for j in range(0, AL.shape[1]):
                yTrue = 1 if y[y_i] == i else 0
                AL[i,j] -= yTrue

        #average over all samples - DONT DO
        #dAL = AL.sum(axis=1)/AL.shape(1)

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
        #don't need param: dx
        dx = 1 if cached_x >= 0 else 0
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

        """
        Key math:
        Setting from notes:
        Use r instead of l for layer
        Z^r = (W^r)^T * A^(r-1) = affine layer
        A^r = g^r(Z^r) = activated layer
        W^r : layer r-1 -> layer r
        g^r is probably relu, but you can add a g^final = softmax step

        Problem: given dL/dA^r, compute dL/A^r and DL/dW^r. We only need to return
        dL/dW^r, bu computing dL/dA^r enables recursion.

        Implement this update:
        Temp: Z^r = g(A^(r-1))
        Set: dL/dA^(r-1) = (W^r)^T * dL/dA^r * g'(Z^r) (note g' not g)
        Set: dL/dW^r = dL/dA^r * g'(Z^r) * (A^(r-1)^)T (note g' not g)

        """

        gradients = []

        dAL_dA_r = dAL
        g = self.activation
        g_prime = self.relu_derivative
        W = self.parameters['weights']

        # start with last layer, note that range(3,-1,-1) == (3, 2, 1, 0)
        for r in range(self.num_layers - 2, -1, -1):
            A_rprev = cache[r-1]
            Z_r =self.activation(cache[r-1])
            dL_dA_rprev = W[r].T * dL_dA_r * g_prime(Z_r)
            dL_dW_r  = dL_dA_r * g_prime(Z_r) * A_rprev.T
            gradients[r] = dL_dW_r

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
        #assuming gradients have been averaged across samples already
        for i,gradient in gradients:
            deltaW = gradient['weights'] * alpha
            self.parameters['weights'][i] -= deltaW
            deltab = gradient['biases'] * alpha
            self.parameters['biases'][i] -= deltab

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
            X_batch, y_batch = self.get_batch(X, y, batch_size)
            # forward prop
            AL, cache = self.forwardPropagation(X_batch)
            # compute loss
            cost, dAL = self.costFunction(AL, y_batch)
            # compute gradients
            gradients = self.backPropagation(dAL, y_batch, cache)
            # update weights and biases based on gradient
            self.updateParameters(gradients, alpha)
            if i % print_every == 0:
                print("Cost: " + cost)
                #self.predict()
                # print cost, train and validation set accuracies

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

        sample = random.sample(xrange(X.shape[1]), batch_size)

        #Assuming X and y are numpy arrays
        X_batch = X[sample]
        y_batch = y[sample]
        return X_batch, y_batch


