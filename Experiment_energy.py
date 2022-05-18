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
    
    
# -----GLOBAL VARIBLES-----
datasetName = "energy-"
S = [8, 6, 5, 2]
funcArray = []
learningRate = 0.2
iters = 5
epochs = 100
percentage = 0.9
epochs = 100
X = energy_X
Y = energy_Y

def main(argv):

    # ================ EXP LEARNING RATE ========================
    if (int(sys.argv[1]) == 1 or sys.argv[1]== 'ALL' or int(sys.argv[1]) >= 1):
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
    
    if (int(sys.argv[2]) == 1 or sys.argv[1]== 'ALL' or int(sys.argv[1]) >=2 ):
        expName = datasetName + "exp_cn"
        experiments = []

        experiment_setups = [[8,8,8,2,2], [8,8,6,4,2], [8,6,6,2,2], [8,4,4,4,2], [8,2,2,2,2],
                            [8,8,8,2], [8,7,4,2], [8,6,6,2], [8,4,4,2], [8,2,2,2],
                            [8,8,2], [8,5,2], [8,6,2], [8,4,2], [8,2,2]]

        for setup in experiment_setups:
            validation = CrossValidation(X, Y, percentage, setup, funcArray, learningRate, iters, epochs)
            accuracy, assertRatio = validation.test()
            expResults = ff.stringify([learningRate, percentage, iters, epochs, accuracy, assertRatio, setup, []])
            experiments.append(expResults)

        ff.store(expName, experiments)

    # ================ EXP: FUNCIONES DE ACTIVACION ========================

    if (int(sys.argv[3]) == 1 or sys.argv[1]== 'ALL' or int(sys.argv[1]) >=3 ):
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

    if (int(sys.argv[4]) == 1 or sys.argv[1]== 'ALL' or int(sys.argv[1]) >=4 ):
            
        expName = datasetName + "exp_ep"
        experiments = []

        epochs = [50, 100, 200, 300, 400, 500]

        for exp_epoch in epochs:
            validation = CrossValidation(X, Y, percentage, S, funcArray, learningRate, iters, exp_epoch)
            accuracy, assertRatio = validation.test()
            expResults = ff.stringify([learningRate, percentage, iters, exp_epoch, accuracy, assertRatio, S, funcArray])
            experiments.append(expResults)

        ff.store(expName, experiments)

