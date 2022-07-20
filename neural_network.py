import sys
import os.path
import numpy as np
import FileFunctions as ff
from Model import Model
from matplotlib import pyplot as plt
import json


""" 
    This python file can be used to run the model. 
    It can be used to train the model or to test the model.

    The file is called as follows:
    python3 neural_network.py  [model_file]  [data_file]  [flag]

    If the model file is not specified, the model will be trained with the data file.

"""

arguments = sys.argv


# ===================== MAIN =====================

def main():

    if (os.path.exists(arguments[0])):    # if model file exists
        # load model from json file
    
        test_data()

    else:                                 # if model file does not exist
        train_with_data()
        test_data()

# ========================== TRAINING =========================


def getData():

    if(arguments[2] == "--energy"):
        X = np.genfromtxt(arguments[1], delimiter=",", dtype=float)[:, :-2]
        Y = np.genfromtxt(arguments[1], delimiter=",", dtype=float)[:, -2:]
        S = [8, 8, 7, 2]
        lr = 0.15
    else:
        X = np.genfromtxt(arguments[1], delimiter=",", dtype=float)[:, :-1]
        Y = np.genfromtxt(arguments[1], delimiter=",", dtype=float)[:, -1:]
        S = [10, 10, 1]
        lr = 0.35

    X = (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0))  # datos de entrada normalizados
    Y = (Y - np.min(Y, axis=0))/(np.max(Y, axis=0) - np.min(Y, axis=0))  # datos de salida normalizados

    model = Model(S, lr, 1000)

    return model, X, Y

# -----------------------------------------------------------

def test_data():
    model, X, Y = getData()

    model.W = [np.load(arguments[0])]

    model.predict(X)

# -----------------------------------------------------------

def train_with_data():
    model, X, Y = getData()

    # train
    model.train(X, Y)

    # save
    np.save(arguments[0], [model.W])

# ====================================================================

if (__name__ == "__main__"):
  main()