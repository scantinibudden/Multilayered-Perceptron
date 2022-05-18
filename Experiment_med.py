from Model import Model
from CrossValidation import CrossValidation
import numpy as np
import FileFunctions as ff
import os

clearConsole = lambda: os.system('clear')

# ================ DATA GET ========================

# -----DATA FOR CANCER DIAGNOSIS PREDICTION-----
# In cancer_Y 1 means 'M' and 0 means 'B'
cancer_X = np.genfromtxt("data/tp1_ej1_training.csv", delimiter=",", dtype=float)[:, 1:]
cancer_Y = np.where(np.genfromtxt("data/tp1_ej1_training.csv", delimiter=",", dtype=str)[:, 0:1] == 'M', 1, 0)[:, :1]

# -----GLOBAL VARIBLES-----
datasetName = "cancer-"
S = [10, 6, 5, 1]
funcArray = []
learningRate = 0.2
iters = 5
epochs = 100
percentage = 0.9
epochs = 100
X = cancer_X
Y = cancer_Y

# ================ EXP LEARNING RATE ========================

expName = datasetName + "exp_lr"
experiments = []

learningRates = [0.01,0.05,0.1,0.2,0.3]

# Experimento medico
for exp_lr in learningRates:
    validation = CrossValidation(X, Y, percentage, S, funcArray, exp_lr, iters, epochs)
    accuracy, assertRatio = validation.test()
    expResults = ff.stringify([exp_lr, percentage, iters, epochs, accuracy, assertRatio, S, funcArray])
    experiments.append(expResults)

ff.store(expName, experiments)

# ================ EXP: CAPAS Y NEURONAS ========================

expName = datasetName + "exp_cn"
experiments = []

experiment_setups = [[10,10,8,2,1], [10,8,6,4,1], [10,6,6,2,1], [10,4,4,4,1], [10,2,2,2,1],
                     [10,10,8,1], [10,7,4,1], [10,6,6,1], [10,4,4,1], [10,2,2,1],
                     [10,8,1], [10,5,1], [10,6,1], [10,4,1], [10,2,1]]

for setup in experiment_setups:
    validation = CrossValidation(X, Y, percentage, setup, funcArray, learningRate, iters, epochs)
    accuracy, assertRatio = validation.test()
    expResults = ff.stringify([learningRate, percentage, iters, epochs, accuracy, assertRatio, setup, []])
    experiments.append(expResults)

ff.store(expName, experiments)

# ================ EXP: FUNCIONES DE ACTIVACION ========================

expName = datasetName + "exp_fa"
experiments = []

functions = [["step", "step", "step"], ["tahn", "tahn", "tahn"], ["sigmoid", "sigmoid", "sigmoid"]]

# Experimento medico
for exp_funcArray in functions:
    validation = CrossValidation(X, Y, percentage, S, exp_funcArray, learningRate, iters, epochs)
    accuracy, assertRatio = validation.test()
    expResults = ff.stringify([learningRate, percentage, iters, epochs, accuracy, assertRatio, S, exp_funcArray])
    experiments.append(expResults)
    
ff.store(expName, experiments)

# ================ EXP: EPOCAS =================

expName = datasetName + "exp_ep"
experiments = []

epochs = [50, 100, 200, 300, 400, 500]

for exp_epoch in epochs:
    validation = CrossValidation(X, Y, percentage, S, funcArray, learningRate, iters, exp_epoch)
    accuracy, assertRatio = validation.test()
    expResults = ff.stringify([learningRate, percentage, iters, exp_epoch, accuracy, assertRatio, S, funcArray])
    experiments.append(expResults)

ff.store(expName, experiments)

