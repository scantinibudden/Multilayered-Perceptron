import numpy as np
import FileFunctions as ff
from Model import Model
from matplotlib import pyplot as plt

cancer_X = np.genfromtxt("data/tp1_ej1_training.csv", delimiter=",", dtype=float)[:, 1:]
cancer_Y = np.where(np.genfromtxt("data/tp1_ej1_training.csv", delimiter=",", dtype=str)[:, 0:1] == 'M', 1, 0)[:, :1]


datasetName = "cancer"

X = cancer_X
Z = cancer_Y
X = (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0)) #datos normalizados

plot_name_template = "results/results_#exp_name#_plot_#run_number#.png"
exp_name = "tp1-exp"

name_template = "results/results_#exp_name#plot#run_number#"
S = [10, 10, 1]

for lr in [0.35]:
    exp_name = f'tp1-lr-tillconv-{lr}'
    model = Model(S, learningRate = lr, maxIter = 5000)

    iters, error, learning = model.train(X, Z)
    exp_info = [model.S, model.activationFuncArray, model.learningRate, iters, model.maxIter, error]
    run_number = ff.store(exp_name, exp_info)

    name = name_template.replace("#exp_name#", exp_name)
    name = name.replace("#run_number#", run_number)
    plot_name = name + ".png"
    csv_name = name + ".csv"

    plt.plot(learning, label=f"Lr = {lr}")
    plt.title("Error evolution")
    plt.ylabel("Estimation error")
    plt.ylim((0, 1))
    plt.xlabel("Epoch")
    plt.legend() 

    plt.savefig(plot_name)