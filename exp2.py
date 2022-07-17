import numpy as np
import FileFunctions as ff
from Model import Model
from matplotlib import pyplot as plt


energy_X = np.genfromtxt("data/tp1_ej2_training.csv", delimiter=",", dtype=float)[:, :-2]
energy_Y = np.genfromtxt("data/tp1_ej2_training.csv", delimiter=",", dtype=float)[:, -2:]

datasetName = "cancer"

X = energy_X
Z = energy_Y

plot_name_template = "results-energy/results_#exp_name#_plot_#run_number#.png"

name_template = "results-energy/results_#exp_name#plot#run_number#"
configs = [[8, 6, 2],]
arrayFunc = ["relu", "relu"]

for S in configs:
    exp_name = f'tp1-energy-config-{S}'
    model = Model(S, activationFuncArray=arrayFunc, learningRate = 0.1, maxIter = 2500)

    iters, error, learning = model.train(X, Z)
    exp_info = [model.S, model.activationFuncArray, model.learningRate, iters, model.maxIter, error]
    run_number = ff.store(exp_name, exp_info)

    name = name_template.replace("#exp_name#", exp_name)
    name = name.replace("#run_number#", run_number)
    plot_name = name + ".png"
    csv_name = name + ".csv"

    plt.plot(learning, label=f"Lr = {S}")
    plt.title("Error evolution")
    plt.ylabel("Estimation error")
    plt.ylim((0, 25))
    plt.xlabel("Epoch")
    plt.legend() 

    plt.savefig(plot_name)