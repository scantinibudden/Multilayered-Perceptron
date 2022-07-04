from CrossValidation import CrossValidation
import numpy as np
import FileFunctions as ff


# ================ DATA GET ========================

# -----DATA FOR CANCER DIAGNOSIS PREDICTION-----
# -----VARIBLES-----
# In cancer_Y 1 means 'M' and 0 means 'B'
cancer_X = np.genfromtxt("data/tp1_ej1_training.csv", delimiter=",", dtype=float)[:, 1:]
cancer_Y = np.where(np.genfromtxt("data/tp1_ej1_training.csv", delimiter=",", dtype=str)[:, 0:1] == 'M', 1, 0)[:, :1]

datasetName = "cancer"

X = cancer_X
Y = cancer_Y
X = (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0)) #datos normalizados

experiments = []
max_epochs =500


# EXPERIMENT 1 =====================================


S = [ X.shape[1] , 5,Y.shape[1]]



funcArray = []                                                     
learningRates = [0.03, 0.5, 0.7]
epsilon = 0.2
iters = 3
percentage = 0.85

for lr in learningRates:
    validation = CrossValidation(X, Y, percentage, S, funcArray, lr, iters, epsilon, max_epochs)
    meanError, epochs, time = validation.test()
    expResults = ff.stringify([datasetName,lr, percentage, iters, epochs, meanError[0], S, funcArray, max_epochs, time])
    experiments.append(expResults)
    

#----------------------------------------



S = [ X.shape[1] , 10,Y.shape[1]] 


funcArray = []                                                     
learningRates = [0.03, 0.5, 0.7]
epsilon = 0.2
iters = 3
percentage = 0.85

for lr in learningRates:
    validation = CrossValidation(X, Y, percentage, S, funcArray, lr, iters, epsilon, max_epochs)
    meanError, epochs, time = validation.test()
    expResults = ff.stringify([datasetName, lr, percentage, iters, epochs, meanError[0], S, funcArray, max_epochs, time])
    experiments.append(expResults)

#----------------------------------------


ff.store(expName, experiments)