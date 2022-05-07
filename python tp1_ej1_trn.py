# ========================= INCLUDES =========================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ========================= SETTINGS =======================

S = [2, 3]        # array with number of neurons per layer
L = len(S)              # number of layers
W = []                  # array of weight matrices
dY = []                 # array of derivatives of the output of each layer
r = 0.01                # learning rate

# ========================= THINGS =========================

def feedForward(X, activationFunc):                 # Xh is what we're trying to predict
    Y = []                                          # Y is the output of each layer
    T = X                                           # T is the input of the next layer
    for k in range(1, L-1):                         # k is the current layer index
        Y.append(addBias(T))                        # add bias to the input
        T = activationFunc(np.dot(Y[k-1], W[k]))    # calculate the output of the layer
    Y.append(T)                                     # add the output of the last layer 
    return Y                                        # return the output of each layer

def backPropagation(Y, Z):                          # Y are the activation levels and Z are the desired output
    dW = [None] * L                                 # delta W is the change in the weight matrix
    E = Z - Y[L-1]                                  # error is the difference between the desired output and the output of the last layer
    D = np.array(L, dtype = float)                  # initialization of D as an empty array
    D[L-1] = E * dY[L-1]                            # dY is the derivative of the output of L-1 layer, and D[L-1] is the product of E and dY
    for k in range(L-1, 1):                         # k is the current layer index
        dW[k] = r * np.dot(Y[k-1].T, D[k])          # dW is the change in the weight matrix, r is the learning rate
        E = np.dot(D[k], W[k].T)                    # E is the error of the next layer = dW * dY
        D[k-1] = subBias(E * dY[L-1-k])             # dY is the derivative of the output of L-1 layer, and D[L-1] is the product of E and dY
    return dW                                       # return the change in the weight matrix
    
# ========================= ACTIVATION FUNCTIONS AND THEIR DERIVATIVES =========================
# EXAMPLE: x = sigmoid(something) => x[0] is the output and x[1] is the derivative

# sigmoid activation function
def sigmoid(x):                                    
    s=1/(1+np.exp(-x))
    ds=s*(1-s)  
    return s,ds

# tanh activation function
def tanh(x):                                        
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=1-t**2
    return t,dt

# relu activation function
def relu(x):                                       
    t = [max(0, value) for value in x]
    dt = [1 if value > 0 else 0 for value in x]
    t = np.array(t, dtype = float)
    dt = np.array(dt, dtype = float)
    return t,dt

# softmax activation function
def softmax(x):                                     
    softmax = np.exp(x) / np.sum(np.exp(x))
    softmax = np.reshape(softmax, (1, -1))
    d_softmax = (softmax * np.identity(softmax.size)- softmax.transpose() @ softmax)
    return softmax,d_softmax

# ========================= AUXILIARY =========================

# adaptation function
def adaptation(X, Y): 
    for k in range(1, L-1):
        X[k] = X[k] + Y[k]
    return X

# estimation function
def estimation(X, Y): 
    Z = X - Y
    return np.inner(Z, Z)

# add bias to the input
def addBias(X):
    return np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)

# remove bias from the input
def subBias(X):
    return X.pop()

# generate random weights
def createRandomWeights(X):
    w = []
    initialW = generateWeights(X.shape[1], S[0])                # first weight matrix
    w.append(initialW)                                   # array of weight matrices
    for i in range(0, L - 1):
        layerW = generateWeights(S[i], S[i + 1])
        w.append(layerW)
    return w

def generateWeights(n, m):
    return np.matrix(np.random.uniform(-1, 1, (n, m)))

X = np.array([[1,1],
     [1,-1],
     [-1,1],
     [-1,-1]])

print(createRandomWeights(addBias(X)))