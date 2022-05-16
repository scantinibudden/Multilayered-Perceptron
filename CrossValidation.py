from Model import Model
import numpy as np

class CrossValidation():
    # ================ CONSTRUCTOR ========================
    def __init__(self, X, Y, trainPercentage, S, funcArray = [], learningRate = 0.1, iters = 100, epoch = 250):
        self.X = X
        self.Y = Y
        self.percentage = trainPercentage
        self.indexes = range(X.shape[0])
        self.S = S
        self.learningRate  = learningRate
        self.funcArray = funcArray
        self.iters = iters
        self.epoch = epoch

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
        accuracies = []
        assertPerc = []
        for i in range(self.iters):
            x_train, x_test, y_train, y_test = self.split()
            model = Model(self.S, self.funcArray, self.learningRate)
            for j in range(self.epoch):
                y = model.train(x_train, y_train)
            y_pred = model.predict(x_test)
            acc = self.accuracy(y_pred, y_test)
            accuracies.append(acc)
            assPerc = self.assertPercentaje(y_pred, y_test)
            assertPerc.append(assPerc)
        return np.mean(np.array(accuracies), axis=0), np.mean(np.array(assertPerc), axis=0)

    def accuracy(self, y_pred, y_check):
        res = np.abs(np.array(y_check) - np.array(y_pred))
        return res.sum(axis=0)/len(y_pred)

    def assertPercentaje(self, y_pred, y_check):
        res = np.abs(np.array(y_check) - np.array(y_pred))
        res = np.where(res == 0, 1, 0)
        return res.sum(axis=0)/len(y_pred)
