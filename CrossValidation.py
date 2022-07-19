from statistics import mean
from Model import Model
import numpy as np
import random
import timeit


class CrossValidation():
    # ================ CONSTRUCTOR ========================
    def __init__(self, X, Y, trainPercentage, S, funcArray=[], learningRate=0.1, iters=100, epsilon=0.1, max_epochs=1):
        self.X = X
        self.Y = Y
        self.percentage = trainPercentage
        self.indexes = range(X.shape[0])
        self.S = S
        self.learningRate = learningRate
        self.funcArray = funcArray
        self.iters = iters
        self.epsilon = epsilon
        self.max_epochs = max_epochs

    # ================ FUNCTIONS ========================
    def split(self):
        index = np.random.choice(self.indexes, int(
            len(self.indexes)*self.percentage), replace=False)
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
        maxDec = 15

        for i in range(self.iters):

            x_train, x_test, y_train, y_test = self.split()
            model = Model(self.S, self.funcArray, self.learningRate)
            # for j in range(self.epoch):
            epoch = 0

            y_pred = model.predict(x_test)
            lastMeanError = self.meanError(y_pred, y_test)
            start = timeit.default_timer()

            while(True):  # this is the way to emulate a do while in python
                epoch = epoch + 1
                x, y, learning = model.train(x_train, y_train)
                y_pred = model.predict(x_test)
                meanError = self.meanError(y_pred, y_test)
                print(
                    f"Epoch: {epoch} with error:{meanError} with mean: {np.mean(meanError)} in Iter {i}")

                if (np.mean(meanError) < self.epsilon or self.max_epochs < epoch):
                    break  # break on the error

            meanErrors.append(meanError)
            epochs.append(epoch)
            times.append(start - timeit.default_timer())
        return np.mean(np.array(meanErrors), axis=0), np.mean(np.array(epochs)),  np.mean(np.array(times), learning)

    def shuffleBoth(self, x_train, y_train):
        temp = list(zip(x_train, y_train))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        return res1, res2

    def meanError(self, y_pred, y_check):
        res = np.abs(np.array(y_check) - np.array(y_pred))
        return res.sum(axis=0)/len(y_pred)
