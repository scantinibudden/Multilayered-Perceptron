from Model import Model
from CrossValidation import CrossValidation
import numpy as np
import FileFunctions as ff

# ================ DATA GET ========================

# -----DATA FOR CANCER DIAGNOSIS PREDICTION-----
# In cancer_Y 1 means 'M' and 0 means 'B'
cancer_X = np.genfromtxt("data/tp1_ej1_training.csv", delimiter=",", dtype=float)[:, 1:]
cancer_Y = np.where(np.genfromtxt("data/tp1_ej1_training.csv", delimiter=",", dtype=str)[:, 0:1] == 'M', 1, 0)[:, :1]

def main(argv):

    # -----VARIBLES-----
    datasetName = "cancer"
    funcArray = []                                                      # vacio corre utilizando sigmoide
    learningRates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    epochs = range(1000, 3001, 500)                                     # [1000, 1500, 2000, 2500, 3000]
    
    iters = 3
    percentage = 0.85
    
    X = cancer_X
    Y = cancer_Y
    X = (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0)) #datos normalizados

    expName = datasetName
    experiments = []
    for ep in epochs:
        for i in range(4):                                              # 1-3 capas internas
            S = [None] * (i+2)
            S[0] = X.shape[1]
            S[-1] = Y.shape[1]
            for j in range(5):
                for k in range(1, i+1):
                    S[k] = np.random.randint(2, 9)
                for lr in learningRates:
                    validation = CrossValidation(X, Y, percentage, S, funcArray, lr, iters, ep)
                    accuracy, meanError = validation.test()
                    expResults = ff.stringify([lr, percentage, iters, ep, accuracy, meanError, S, funcArray])
                    experiments.append(expResults)
    
    ff.store(expName, experiments)

if __name__ == "__main__":
    main(sys.argv)