from ast import arg
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
    global arguments
    if len(arguments) == 1:

        arguments = ["mew uwu mew", "model_med.npy",
                     "data/tp1_ej1_training.csv", "--med"]

        # arguments = ["mew uwu mew", "model_energy.npy",
        #              "data/tp1_ej2_training.csv", "--energy"]

    if (not os.path.exists(arguments[1])):    # if model file exists

        train_with_data()

    test_data()

# ========================== TRAINING =========================


def getData():

    if(arguments[3] == "--energy"):
        print("Loading energy data...")
        X = np.genfromtxt(arguments[2], delimiter=",", dtype=float)[:, :-2]

        Y = np.genfromtxt(arguments[2], delimiter=",", dtype=float)[:, -2:]
        S = [8, 8, 7, 2]
        lr = 0.15

    elif(arguments[3] == "--med"):
        print("Loading medical data...")
        X = np.genfromtxt(arguments[2], delimiter=",", dtype=float)[:, 1:]
        Y = np.where(np.genfromtxt(arguments[2], delimiter=",", dtype=str)[
                     :, 0:1] == 'M', 1, 0)[:, :1]
        Y.astype(float)
        S = [10, 10, 1]
        lr = 0.35

    X = (X - np.min(X, axis=0))/(np.max(X, axis=0) -
                                 np.min(X, axis=0))  # datos de entrada normalizados
    Y = (Y - np.min(Y, axis=0))/(np.max(Y, axis=0) -
                                 np.min(Y, axis=0))  # datos de salida normalizados

    model = Model(S, [], lr, 10)
    return model, X, Y

# -----------------------------------------------------------


def train_with_data():

    print("Training model...")
    model, X, Y = getData()

    # train
    model.train(X, Y)
    print("Training done.")
    # save
    np.save(arguments[1], model.W)

    print("Model saved as: ", arguments[1])

# -----------------------------------------------------------


def test_data():

    print("Testing model...")
    model, X, Y = getData()
    model.W = np.load(arguments[1], allow_pickle=True)
    # np.load = np_load_old
    print("Model loaded from: ", arguments[1])

    y_pred = model.predict(X)

    print("Prediction done.")

    print_results(y_pred, Y)

# -----------------------------------------------------------


def print_results(y_pred, Y):
    error = Y - y_pred
    meanError = np.mean(np.abs(error), axis=0)
    print("Mean error: ", np.mean(meanError))

# -----------------------------------------------------------


# ====================================================================

if (__name__ == "__main__"):
    main()
