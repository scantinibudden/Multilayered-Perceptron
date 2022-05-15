import pandas as pd
import numpy as np

header_names = ["lr", "splitPercentaje", "iter", "epoch", "accuracy", "assertPerc", "layers", "activationFunctions"]

cancer_df = pd.read_csv("results/results_hola_1.txt", names=header_names)
display(cancer_df)

