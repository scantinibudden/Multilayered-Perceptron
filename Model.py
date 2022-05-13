# ========================= INCLUDES =========================

import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

class Model():
    def __init__(self, X, Z, S, activationFuncArray = [], learningRate = 0.1, maxIter = 100):
    # ========================= VARS =======================

        self.X = X                                                      # input
        self.Z = Z                                                      # expected output
        self.S = S                                                      # array with number of neurons per inner layer
        self.S.insert(0, X.shape[1])
        self.S.append(Z.shape[1])
        self.layers = len(S)                                            # number of layers
        self.W = self.createRandomWeights()                             # array of weight matrices
        self.learningRate = learningRate                                # learning rate
        self.iterations = 0                                             # number of iterations
        if (activationFuncArray == []):                                 # default funcArray uses sigmoid for every layer
            activationFuncArray = ["sigmoid"]*len(S)                    # set default activation array
        self.activationFuncArray = activationFuncArray                  # activation function for each layers
        self.activationFuncDicc = {
            "sigmoid" : self.sigmoid,
            "tahn" : self.tanh,
            "relu" : self.relu,
            "softmax" : self.softmax,
            "escalonada" : self.escalonada
        }
        
        
    # ========================= FUNCTIONS   =========================
    
    def feedForward(self, Xh):                                          # Xh is what we're trying to predict
        Y = []                                                          # Y is the output of each layer
        dY = [0]
        T = np.array(Xh)                                                # T is the input of the next layer
        for k in range(1, self.layers):                                 # k is the current layer index
            func = self.activationFuncDicc[self.activationFuncArray[k-1]] # get activation function
            Y.append(self.addBias(T))                                   # add bias to the input
            T, dT = func(np.dot(Y[k-1], self.W[k-1]))                   # calculate the output of the layer
            dY.append(dT)                                               # save the derivative of the output
        Y.append(T)                                                     # add the output of the last layer 
        return Y, dY                                                    # return the output of each layer

#a = np.empty(n)
    def backPropagation(self, Y, dY, h):                                                                            # Y are the activation levels and Z are the desired output
        dW = [None] * self.layers                                                                                   # delta W is the change in the weight matrix
        E = np.array(self.Z[h] - Y[self.layers - 1])                                                                # error is the difference between the desired output and the output of the last layer
        D = [None] * self.layers                                                                                    # initialization of D as an empty array

        D[self.layers-1] = np.multiply(E, dY[self.layers - 1])                                                                  # dY is the derivative of the output of L-1 layer, and D[layers-1] is the product of E and dY
        for k in range(1, self.layers):                                                                             # k is the current layer index
            j = self.layers - k
            dW[j-1] = self.learningRate*np.dot(np.transpose(np.reshape(Y[j-1], (1,Y[j-1].shape[0]))), D[j])         # dW is the change in the weight matrix
            E = np.dot(D[j], np.transpose(self.W[j-1]))                                                             # E is the error of the next layer = dW * dY
        return dW                                                                                                   # return the change in the weight matrix
    
    def train(self):
        ans = []
        for h in range(len(self.X)):
            Y, dY = self.feedForward(self.X[h])                         # get the output of each layer
            ans.append(np.array(Y[self.layers - 1]))                              # save the output of the last layer
            dW = self.backPropagation(Y, dY, h)
            self.adaptation(dW)
        return ans
        
            
    
    # ========================= ACTIVATION FUNCTIONS AND THEIR DERIVATIVES =========================
    # EXAMPLE: x = sigmoid(something) => x[0] is the output and x[1] is the derivative

    # sigmoid activation function
    def sigmoid(self, x):                                    
        s=1/(1+np.exp(x))
        ds=np.multiply(s, (1-s))
        return s,ds

    # tanh activation function
    def tanh(self, x):                                        
        t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        dt=1-t**2
        return t,dt

    # relu activation function
    def relu(self, x):                                       
        t = [max(0, value) for value in x]
        dt = [1 if value > 0 else 0 for value in x]
        t = np.array(t, dtype = float)
        dt = np.array(dt, dtype = float)
        return t,dt

    # softmax activation function
    def softmax(self, x):                                     
        softmax = np.exp(x) / np.sum(np.exp(x))
        softmax = np.reshape(softmax, (1, -1))
        d_softmax = np.multiply(softmax, np.dot(np.identity(softmax.size) - softmax.transpose(), softmax))
        return softmax,d_softmax

    def escalonada(self, x):
        sign = np.sign(x)
        return sign, np.ones(sign.shape)
            
    # ========================= AUXILIARY =========================

    # adaptation function
    def adaptation(self, dW): 
        for k in range(1, self.layers):       #k es la layer
            self.W[k-1] = self.W[k-1] + dW[k-1]

    # estimation function
    def estimation(self, Y): 
        e = self.Z - Y
        return np.inner(e, e)

    # add bias to the input
    def addBias(self, x):
        return np.append(x, 1)

    # remove bias from the input
    def subBias(self,Y):
        return Y[:,:-1]


    def generateWeights(self, n, m):
        return np.matrix(np.random.uniform(-1, 1, (n, m)))

    # generate random weights
    def createRandomWeights(self):
        w = []                                                       # array of weight matrices
        for i in range(0, self.layers - 1):
            layerW = self.generateWeights(self.S[i] + 1, self.S[i + 1])
            w.append(layerW)
        return w

X = np.array([[1,1],
    [1,-1],
    [-1,1],
    [-1,-1]])

Z = np.array([[1,1],[-1,1],[-1,1],[-1,-1]])

S = []

funcArray = ["sigmoid"] # sigmoid anda pero al reves

model = Model(X, Z, S, funcArray)
for i in range(0, 250):
    y = model.train()
print(y)