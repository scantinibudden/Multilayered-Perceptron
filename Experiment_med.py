from CrossValidation import CrossValidation
import numpy as np
import FileFunctions as ff
import timeit

# ================ DATA GET ========================

# -----DATA FOR CANCER DIAGNOSIS PREDICTION-----
# In cancer_Y 1 means 'M' and 0 means 'B'
cancer_X = np.genfromtxt("data/tp1_ej1_training.csv", delimiter=",", dtype=float)[:, 1:]
cancer_Y = np.where(np.genfromtxt("data/tp1_ej1_training.csv", delimiter=",", dtype=str)[:, 0:1] == 'M', 1, 0)[:, :1]

# -----VARIBLES-----
datasetName = "cancer"
funcArray = []                                                      # vacio corre utilizando sigmoide
# learningRates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# epochs = range(1000, 3001, 500)                                     # [1000, 1500, 2000, 2500, 3000]
lr = 0.05
epsilon = 0.25
iters = 1
percentage = 0.85

X = cancer_X
Y = cancer_Y
X = (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0)) #datos normalizados

expName = datasetName
experiments = []

S = [ X.shape[1] , 5,Y.shape[1]]

batches = [1,2,3]
for batch_size in batches:
    start = timeit.default_timer()
    validation = CrossValidation(X, Y, percentage, S, funcArray, lr, iters, epsilon, batch_size)
    stop = timeit.default_timer()
    meanError, epochs = validation.test()
    expResults = ff.stringify([lr, percentage, iters, epochs, meanError[0], S, funcArray, batch_size, start - stop])
    experiments.append(expResults)
                
ff.store(expName, experiments)