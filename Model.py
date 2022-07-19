# ========================= INCLUDES =========================

import numpy as np


class Model():
    def __init__(self, S=[], activationFuncArray=[], learningRate=0.001, maxIter=1000):
        # ========================= VARS =======================
        # array with number of neurons per inner layer
        self.S = S
        # number of layers
        self.layers = len(S)
        # array of weight matrices
        self.W = self.createRandomWeights()
        self.learningRate = learningRate                                # learning rate
        self.maxIter = maxIter
        # default funcArray uses sigmoid for every layer
        if (activationFuncArray == []):
            # set default activation array
            activationFuncArray = ["sigmoid"]*len(S)
        # activation function for each layers
        self.activationFuncArray = activationFuncArray
        self.activationFuncDicc = {
            "sigmoid": self.sigmoid,
            "tanh": self.tanh,
            "relu": self.relu,
            "step": self.step
        }

    # ========================= FUNCTIONS   =========================

    # Xh is what we're trying to predict
    def feedForward(self, Xh):
        # Y is the output of each layer
        Y = []
        dY = [0]
        # T is the input of the next layer
        T = np.array(Xh)
        # k is the current layer index
        for k in range(1, self.layers):
            # get activation function
            func = self.activationFuncDicc[self.activationFuncArray[k-1]]
            # add bias to the input
            Y.append(self.addBias(T))
            # calculate the output of the layer
            T, dT = func(np.dot(Y[k-1], self.W[k-1]))
            # save the derivative of the output
            dY.append(np.array(dT))
        # add the output of the last layer
        Y.append(np.array(T[0]))
        # return the output of each layer
        return Y, dY

    # Y are the activation levels and Z are the desired output
    def backPropagation(self, Y, dY, dZ):
        # delta W is the change in the weight matrix
        dW = [None] * self.layers
        # error is the difference between the desired output and the output of the last layer
        E = np.array(dZ - Y[self.layers - 1])
        # initialization of D as an empty array
        D = [None] * self.layers
        # dY is the derivative of the output of L-1 layer, and D[layers-1] is the product of E and dY
        D[self.layers-1] = np.multiply(E, dY[self.layers - 1])
        for k in np.flip(range(1, self.layers)):
            # dW is the change in the weight matrix
            dW[k-1] = self.learningRate * \
                np.dot(np.transpose(np.reshape(
                    Y[k-1], (1, Y[k-1].shape[0]))), D[k])
            # E is the error of the next layer = dW * dY
            E = np.dot(D[k], np.transpose(self.W[k-1]))
            D[k-1] = np.multiply(self.subBias(E), dY[k-1])
        # return the change in the weight matrix
        return dW

    def train(self, X, Z):
        iters = 0
        meanError = 1
        learning = []
        while iters < self.maxIter and np.mean(meanError) > 0.01:

            for h in range(0, len(X)):
                Y, dY = self.feedForward(X[h])
                dZ = Z[h]
                dW = self.backPropagation(Y, dY, dZ)
                self.adaptation(dW)

            y_pred = self.predict(X)

            error = Z - y_pred

            meanError = np.mean(np.abs(error), axis=0)
            learning.append(meanError)
            iters += 1
            print(
                f"Mean error: {meanError} and : {np.mean(meanError)} on iter {iters}")

        return iters, meanError, learning

    def predict(self, X):
        ans = []
        for h in range(len(X)):
            # get the output of each layer
            Y, dY = self.feedForward(X[h])
            # save the output of the last layer
            ans.append(np.array(Y[-1][0]))
        return ans

    # ========================= ACTIVATION FUNCTIONS AND THEIR DERIVATIVES =========================
    # EXAMPLE: x = sigmoid(something) => x[0] is the output and x[1] is the derivative

    # sigmoid activation function
    def sigmoid(self, x):
        t = 1 / (1 + np.exp(-x))
        dt = np.multiply(t, 1 - t)
        return t, np.array(dt)

    # tanh activation function
    def tanh(self, x):
        t = np.tanh(x)
        dt = 1 - np.power(t, 2)
        return t, dt

    # relu activation function
    def relu(self, x):
        t = np.maximum(0, x)
        dt = np.where(x > 0, 1, 0)
        return t, dt

    # binary step activation function
    def step(self, x):
        t = np.sign(x)
        t = np.where(x > 0, 1, 0)
        dt = np.ones(t.shape)
        return t, dt

    # ========================= AUXILIARY =========================

    # adaptation function
    def adaptation(self, dW):
        for k in range(1, self.layers):
            self.W[k-1] = self.W[k-1] + dW[k-1]

    # add bias to the input
    def addBias(self, x):
        return np.append(np.array(x), 1)

    # remove bias from the input
    def subBias(self, Y):
        return Y[:, :-1]

    def generateWeights(self, n, m):
        return np.matrix(np.random.uniform(-1, 1, (n, m)))

    # generate random weights
    def createRandomWeights(self):

        # array of weight matrices
        w = []

        for i in range(0, self.layers - 1):
            layerW = self.generateWeights(self.S[i] + 1, self.S[i + 1])
            w.append(layerW)
        return w
