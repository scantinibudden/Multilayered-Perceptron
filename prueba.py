import numpy as np
import FileFunctions as ff
from Model import Model
from matplotlib import pyplot as plt
import json
# ================ DATA GET ========================

# -----DATA FOR ELECTRIC CONSUMPTION ESTIMATION-----
energy_X = np.genfromtxt("data/tp1_ej2_training.csv",
                         delimiter=",", dtype=float)[:, :-2]
energy_Y = np.genfromtxt("data/tp1_ej2_training.csv",
                         delimiter=",", dtype=float)[:, -2:]

# -----VARIBLES-----
datasetName = "energy"
X = energy_X
Z = energy_Y

X = (X - np.min(X, axis=0))/(np.max(X, axis=0) -
                             np.min(X, axis=0))  # datos de entrada normalizados
Z = (Z - np.min(Z, axis=0))/(np.max(Z, axis=0) -
                             np.min(Z, axis=0))  # datos de salida normalizados


# -------------------------------------
S = [X.shape[1], 8, Z.shape[1]]
model = Model(S,
              learningRate=0.3,
              maxIter=10)

iters, error, learning = model.train(X, Z)

# data = {}
# data["model"] = model.W
# with open("model.json", "w") as write_file:
# json.dump(data, write_file)
print(model.W)
np.save("model.npy", model.W[0])
