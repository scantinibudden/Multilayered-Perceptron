from Model import Model
import numpy as np
import random
import timeit

class CrossValidation():
    # ================ CONSTRUCTOR ========================
    def __init__(self, X, Y, trainPercentage, S, funcArray = [], learningRate = 0.1, iters = 100, epsilon = 0.1, batch_size = 1):
        self.X = X
        self.Y = Y
        self.percentage = trainPercentage
        self.indexes = range(X.shape[0])
        self.S = S
        self.learningRate  = learningRate
        self.funcArray = funcArray
        self.iters = iters
        self.epsilon = epsilon
        self.batch_size = batch_size

    # ================ FUNCTIONS ========================
    def split(self):
        index = np.random.choice(self.indexes, int(len(self.indexes)*self.percentage), replace = False)
        not_index = np.setdiff1d(self.indexes, index)
        x_train = [self.X[i] for i in index]
        x_test = [self.X[i] for i in not_index]
        y_train = [self.Y[i] for i in index]
        y_test = [self.Y[i] for i in not_index]
        return x_train, x_test, y_train, y_test

    def test(self):
        meanErrors = []
        epochs = []
        times = []
        maxDec = 20

        for i in range(self.iters):

            x_train, x_test, y_train, y_test = self.split()
            model = Model(self.S, self.funcArray, self.learningRate)
            #for j in range(self.epoch):
            epoch = 0
            batch_size = self.batch_size
            n=0

            y_pred = model.predict(x_test)
            lastMeanError = self.meanError(y_pred, y_test)
            start = timeit.default_timer()
            while(True): # this is the way to emulate a do while in python
                epoch = epoch +1
                model.train(x_train, y_train, batch_size)
                y_pred = model.predict(x_test)
                meanError = self.meanError(y_pred, y_test)
                print(f"Epoch: {epoch} with error:{meanError} in Iter {i}")
                if (meanError > lastMeanError):
                    n +=1
                else: 
                    n = 0
                lastMeanError = meanError
                if (n == maxDec):
                    #x_train, y_train = self.shuffleBoth(x_train, y_train)
                    n = 0
                if (epoch % 300 == 0): 
                    x_train, y_train = self.shuffleBoth(x_train, y_train)

                if (meanError < self.epsilon): break #break on the error

            meanErrors.append(meanError)
            epochs.append(epoch)
            times.append(start - timeit.default_timer())
        return np.mean(np.array(meanErrors), axis=0), np.mean(np.array(epochs)),  np.mean(np.array(times))

    def shuffleBoth(self,x_train, y_train):
        temp = list(zip(x_train, y_train))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        return res1, res2

    def meanError(self, y_pred, y_check):
        res = np.abs(np.array(y_check) - np.array(y_pred))
        return res.sum(axis=0)/len(y_pred)
