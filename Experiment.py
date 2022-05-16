from Model import Model
from CrossValidation import CrossValidation
import numpy as np
import FileFunctions as ff


# ================ DATA GET ========================

# -----DATA FOR CANCER DIAGNOSIS PREDICTION-----
# In cancer_Y 1 means 'M' and 0 means 'B'
#cancer_X = np.genfromtxt("data/tp1_ej1_training.csv", delimiter=",", dtype=float)[:, 1:]
#cancer_Y = np.where(np.genfromtxt("data/tp1_ej1_training.csv", delimiter=",", dtype=str)[:, 0:1] == 'M', 1., 0.)[:, :1]

# -----DATA FOR ELECTRIC CONSUMPTION ESTIMATION-----
#electric_X = np.genfromtxt("data/tp1_ej2_training.csv", delimiter=",", dtype=float)[:, :-2]
#electric_Y = np.genfromtxt("data/tp1_ej2_training.csv", delimiter=",", dtype=float)[:, -2:]

# ================ EJEMPLO DE JUGUETE (calvo) ========================

# X = cancer_X
# Z = cancer_Y
# perc = 0.8
# S = []
# funcArray = []
# lr =
# iters =
# epoch =

# validation = CrossValidation(X, Z, perc, S, funcArray, lr, iters, epoch)
#==================================
expName = "test recontra juguete"
X = np.array([[1,1],
    [1,-1],
    [-1,1],
    [-1,-1],
    [1,1],
    [1,-1],
    [-1,1],
    [-1,-1],
    [1,1],
    [1,-1],
    [-1,1],
    [-1,-1],
    [1,1],
    [1,-1],
    [-1,1],
    [-1,-1],
    [1,1],
    [1,-1],
    [-1,1],
    [-1,-1],
    [1,1],
    [1,-1],
    [-1,1],
    [-1,-1]])

Z = np.array([[1,1],[-1,1],[-1,1],[-1,-1],[1,1],[-1,1],[-1,1],[-1,-1],[1,1],[-1,1],[-1,1],[-1,-1],[1,1],[-1,1],[-1,1],
              [-1,-1],[1,1],[-1,1],[-1,1],[-1,-1],[1,1],[-1,1],[-1,1],[-1,-1]])

splitPerc = 0.8

S = [2, 2, 2, 2]

funcArray = ["step", "step", "step"]

iters = 20
epochs = 250
lr = 0.2

experiments = []
for i in range(1, 2):

    experimento = CrossValidation(X, Z, splitPerc, S, funcArray, lr, iters, epochs)
    accuracy, assertRatio = experimento.test()
    expResults= ff.stringify([lr, splitPerc, iters, epochs, accuracy, assertRatio, S, funcArray])
    experiments.append(expResults)

ff.store(expName, experiments)

#header_names = ["lr", "splitPercentaje", "iter", "epoch", "accuracy", "assertPerc", "layers", "activationFunctions"]


# ============================== EXP RANDOMS =======================================

# exp_name = "exp_random"
# data_name = "medic"
# iter = 20
# tests = []
# for i in range(0, iter):
#     layers = i
#     S = []
#     for j in range(0, layers):
#         neurons = np.random.random(10) + 1
#         S.append(neurons)
#     lr = i*1/iter + 0.01
#     funcArray = np.repeat(["step"], layers + 1)
#     epoch = np.random.random(500) + 1
#     perc = np.random.random(1)
#     validation = CrossValidation(X, Z, perc, funcArray, lr, 500, epoch)
#     assertP, accuracy = validation.test()

#     test = {
#         "learningRate"  : lr,
#         "splitPercentage"  : perc,
#         "iter"  : 500,
#         "epochs"  : epoch,
#         "accuracy"  : accuracy,
#         "assertPercentage"  : assertP,
#         "layers"  : S,
#         "activationFunctions"  : funcArray}
#     tests.append(test)


# lrs = []
# for i in range(0, len(tests)):
#     lrs.append(tests[i]["learningRate"])

# a = np.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
# np.savetxt("foo.csv", a, delimiter=",")
