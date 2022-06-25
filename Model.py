# ========================= INCLUDES =========================

import numpy as np

class Model():
    def __init__(self, S = [], activationFuncArray = [], learningRate = 0.1):
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
            "tanh" : self.tanh,
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
            dY.append(np.array(dT))                                               # save the derivative of the output
        Y.append(np.array(T[0]))                                        # add the output of the last layer
        return Y, dY                                                    # return the output of each layer

    def backPropagation(self, Y, dY, Z, h):                                                                         # Y are the activation levels and Z are the desired output
        dW = [None] * self.layers                                                                                   # delta W is the change in the weight matrix
        E = np.array(Z[h] - Y[self.layers - 1])                                                                     # error is the difference between the desired output and the output of the last layer
        D = [None] * self.layers                                                                                    # initialization of D as an empty array
        D[self.layers-1] = np.multiply(E, dY[self.layers - 1])                                                      # dY is the derivative of the output of L-1 layer, and D[layers-1] is the product of E and dY
        for k in np.flip(range(1, self.layers)):
            dW[k-1] = self.learningRate*np.dot(np.transpose(np.reshape(Y[k-1], (1,Y[k-1].shape[0]))), D[k])         # dW is the change in the weight matrix
            E = np.dot(D[k], np.transpose(self.W[k-1]))                                                             # E is the error of the next layer = dW * dY
            D[k-1] = np.multiply(self.subBias(E), dY[k-1])
        return dW                                                                                                   # return the change in the weight matrix
    
    def train(self, X, Z):
        ans = []
        meanY, meanDY # declaranding
        batch_size = 100
        for h in range(1, len(X)):

            if(h % 100 == 0):
                meanY, meanDY = self.feedForward(X[h])
            else:
                meanY, meanDY += self.feedForward(X[h])                                      # get the output of each layer
            ans.append(np.array(meanY[-1]))                                         # save the output of the last layer
            if(h % batch_size == batch_size - 1 or h == len()-1):
                meanY, meanDY /= (h % batch_size) + 1
                dW = self.backPropagation(meanY, meanDY, Z, h)
                self.adaptation(dW)
        return ans
        
    def predict(self, X):
        ans = []
        for h in range(len(X)):
            Y, dY = self.feedForward(X[h])                                      # get the output of each layer
            ans.append(np.array(Y[-1][0]))                                         # save the output of the last layer
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