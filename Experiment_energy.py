from Model import Model
from CrossValidation import CrossValidation
import numpy as np
import FileFunctions as ff
import os
import sys
clearConsole = lambda: os.system('clear')

# ================ DATA GET ========================
# -----DATA FOR ELECTRIC CONSUMPTION ESTIMATION-----
energy_X = np.genfromtxt("data/tp1_ej2_training.csv", delimiter=",", dtype=float)[:, :-2]
energy_Y = np.genfromtxt("data/tp1_ej2_training.csv", delimiter=",", dtype=float)[:, -2:]

# -----VARIBLES-----
datasetName = "energy"
funcArray = []                                                      # vacio corre utilizando sigmoide
learningRates = [0.05, 0.1, 0.15]
epochs = range(2000, 3001, 500)                                     # [2000, 2500, 3000]

iters = 1
percentage = 0.85

X = energy_X
Y = energy_Y

X = (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0)) #datos de entrada normalizados
Y = (Y - np.min(Y, axis=0))/(np.max(Y, axis=0) - np.min(Y, axis=0)) #datos de salida normalizados

expName = datasetName
experiments = []
for ep in epochs:
    for i in range(0, 2):                                              # 1-3 capas internas
        S = [None] * (i+2)
        S[0] = X.shape[1]
        S[-1] = Y.shape[1]
        for k in range(1, i+1):
            S[k] = np.random.randint(3, 7)
        for lr in learningRates:
            validation = CrossValidation(X, Y, percentage, S, funcArray, lr, iters, ep)
            accuracy, meanError = validation.test()
            expResults = ff.stringify([lr, percentage, iters, ep, accuracy, meanError, S, funcArray])
            experiments.append(expResults)
            print(len(experiments))

ff.store(expName, experiments)
