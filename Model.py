# ========================= INCLUDES =========================

import numpy as np

class Model():
    def __init__(self, S, activationFuncArray = [], learningRate = 0.1):
    # ========================= VARS =======================
        self.S = S                                                      # array with number of neurons per inner layer
        self.layers = len(S)                                            # number of layers
        self.W = self.createRandomWeights()                             # array of weight matrices
        self.learningRate = learningRate                                # learning rate
        if (activationFuncArray == []):                                 # default funcArray uses sigmoid for every layer
            activationFuncArray = ["sigmoid"]*len(S)                    # set default activation array
        self.activationFuncArray = activationFuncArray                  # activation function for each layers
        self.activationFuncDicc = {
            "sigmoid" : self.sigmoid,
            "tahn" : self.tanh,
            "relu" : self.relu,
            "step" : self.step
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
    def backPropagation(self, Y, dY, Z, h):                                                                            # Y are the activation levels and Z are the desired output
        dW = [None] * self.layers                                                                                   # delta W is the change in the weight matrix
        E = np.array(Z[h] - Y[self.layers - 1])                                                                # error is the difference between the desired output and the output of the last layer
        D = [None] * self.layers                                                                                    # initialization of D as an empty array

        D[self.layers-1] = np.multiply(E, dY[self.layers - 1])                                                                  # dY is the derivative of the output of L-1 layer, and D[layers-1] is the product of E and dY
        for k in range(1, self.layers):                                                                             # k is the current layer index
            j = self.layers - k
            dW[j-1] = self.learningRate*np.dot(np.transpose(np.reshape(Y[j-1], (1,Y[j-1].shape[0]))), D[j])         # dW is the change in the weight matrix
            E = np.dot(D[j], np.transpose(self.W[j-1]))                                                             # E is the error of the next layer = dW * dY
            D[j-1] = np.multiply(self.subBias(E), dY[j-1])
        return dW                                                                                                   # return the change in the weight matrix
    
    def train(self, X, Z):
        ans = []
        for h in range(len(X)):
            Y, dY = self.feedForward(X[h])                         # get the output of each layer
            ans.append(np.array(Y[self.layers - 1]))                              # save the output of the last layer
            dW = self.backPropagation(Y, dY, Z, h)
            self.adaptation(dW)
        return ans
        
    def predict(self, X):
        Y = []
        for h in range(len(X)):
            y, dy = self.feedForward(X[h])
            y = np.array(y[self.layers - 1][0])[0]
            Y.append(y)
        return Y
    
    # ========================= ACTIVATION FUNCTIONS AND THEIR DERIVATIVES =========================
    # EXAMPLE: x = sigmoid(something) => x[0] is the output and x[1] is the derivative

    # sigmoid activation function
    def sigmoid(self, x):        
        t = 1 / (1 + np.exp(-x))
        dt = np.multiply(t, 1 - t)
        return t, dt

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
        dt = np.ones(t.shape)
        return t, dt
            
    # ========================= AUXILIARY =========================

    # adaptation function
    def adaptation(self, dW): 
        for k in range(1, self.layers):
            self.W[k-1] = self.W[k-1] + dW[k-1]

    # estimation function
    def estimation(self, Y): 
        e = self.Z - Y
        return np.inner(e, e)

    # add bias to the input
    def addBias(self, x):
        return np.append(np.array(x), 1)

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
