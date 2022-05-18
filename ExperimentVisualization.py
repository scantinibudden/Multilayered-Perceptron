import pandas as pd
import numpy as np
import sys

header_names = ["lr", "splitPercentaje", "iter", "epoch", "error", "accuracy", "layers", "activationFunctions"]

cancer_df = pd.read_csv("results/results_hola_1.txt", names=header_names)
display(cancer_df)
fileType = "txt"
def main(argv):
    if (int(sys.argv[1]) == 1 or sys.argv[1]== 'ALL' or int(sys.argv[1]) >= 1):
        
    # ================ EXP LEARNING RATE ========================
        
        
        # DATA PREPARATION ------------------------
        # + medical experiments
        exp_file = "expblabla"
        columnToWatch = "layers"
        expectedInColumn = "10,8,6,4,2"
        
        
        df_resultados_med = pd.read_csv(f"results/{exp_file}.{fileType}",names=header_names)
        df_resultados_med = df_resultados_med[(df_resultados_med[columnToWatch]==expectedInColumn)]
        df_resultados_med = df_resultados_med.sort_values(by=['lr'])
        
        # + electric experiments
        exp_file = "expblabla"
        columnToWatch = "layers"
        expectedInColumn = "10,8,6,4,2"
        
        
        df_resultados_elec = pd.read_csv(f"results/{exp_file}.{fileType}", names=header_names)
        df_resultados_elec = df_resultados_elec[(df_resultados_elec[columnToWatch]==expectedInColumn)]
        df_resultados_elec = df_resultados_elec.sort_values(by=['lr'])
        
        #PLOT PREPARATION ------------------------
        title = "Experimento sobre learning rate"
        fileName = "exp_1"
        l = "-"
        name1 = "Dataset Medico"
        name2 = "Dataset Electrico"
        xAxis = "lr"
        yAxis = "error"


        fig,(ax1,ax2) = plt.subplots(1,2, figsize=(12,3))
        fig.suptitle(title.title())

        ax1.set_title(name1)
        ax1.set_ylabel("Error")
        ax1.set_xlabel("Lr")
        ax2.set_title(name2)
        ax2.set_ylabel("Error")
        ax2.set_xlabel("Lr")

        ax1.plot(df_resultados_med[xAxis],df_resultados_med[yAxis],l, c="blue", label="Med")
        ax2.plot(df_resultados_elec[xAxis],df_resultados_elec[yAxis],l, c="blue", label="Elec")

        ax1.legend()
        ax2.legend()

        plt.show()
        
        # plt.savefig(fileName+".png")
        
        
    # ---------------------------------------------------------------------------------
        
    if (int(sys.argv[2]) == 1 or sys.argv[1]== 'ALL' or int(sys.argv[1]) >=2 ):
        
    # ================ EXP: CAPAS Y NEURONAS ========================
        
        
        
        
    #SHOULD BE IMPLEMENTED
        
        
        
        
        
        
    
    # ---------------------------------------------------------------------------------
    
    if (int(sys.argv[3]) == 1 or sys.argv[1]== 'ALL' or int(sys.argv[1]) >=3 ):
        
        # CUIDADO: NO ESTA COMPlETO
        
    # ================ EXP: FUNCIONES DE ACTIVACION ========================
        
        # DATA PREPARATION ------------------------
        # + medical experiments
        exp_file = "expblabla"
        columnToWatch = "activationFunctions"
        expectedInColumn = [ "sigmoid","sigmoid-sigmoid","sigmoid-sigmoid-sigmoid"]
        
        
        df_resultados_med = pd.read_csv(f"results/{exp_file}.{fileType}",names=header_names)
        df_resultados_med_1 = df_resultados_med[(df_resultados_med[columnToWatch]==expectedInColumn[0])]
        df_resultados_med_2 = df_resultados_med[(df_resultados_med[columnToWatch]==expectedInColumn[1])]
        df_resultados_med_3 = df_resultados_med[(df_resultados_med[columnToWatch]==expectedInColumn[2])]
        
        # + electric experiments
        exp_file = "expblabla"
        columnToWatch = "layers"
        expectedInColumn = ["tahn", "than-than", "than-than-than"]
        
        
        df_resultados_elec = pd.read_csv(f"results/{exp_file}.{fileType}", names=header_names)
        df_resultados_elec_1 = df_resultados_elec[(df_resultados_elec[columnToWatch]==expectedInColumn[0])]
        df_resultados_elec_2 = df_resultados_elec[(df_resultados_elec[columnToWatch]==expectedInColumn[1])]
        df_resultados_elec_3 = df_resultados_elec[(df_resultados_elec[columnToWatch]==expectedInColumn[2])]
        
        
        
        
        #PLOT PREPARATION ------------------------
        title = "Experimento sobre funciones activacion"
        fileName = "exp_func"
        l = "-"
        name1 = "Dataset Medico"
        name2 = "Dataset Electrico"
        xAxis = "lr"
        yAxis = "error"


        fig,(ax1,ax2) = plt.subplots(1,2, figsize=(12,3))
        fig.suptitle(title.title())

        ax1.set_title(name1)
        ax1.set_ylabel("Error")
        ax1.set_xlabel("Number of Layers")
        ax2.set_title(name2)
        ax2.set_ylabel("Error")
        ax2.set_xlabel("Number of Layers")

        
        ax1.bar(1,df_resultados_elec_1["error"] , label="1 layer")
        ax1.bar(2,df_resultados_elec_2["error"] , label="2 layer")
        ax1.bar(3,df_resultados_elec_3["error"] ,label="3 layer")
        
        ax2.bar(1,df_resultados_med_1["error"] , label="1 layer")
        ax2.bar(2,df_resultados_med_2["error"] , label="2 layer")
        ax2.bar(3,df_resultados_med_3["error"] ,label="3 layer")
        
        
        ax1.legend()
        ax2.legend()

        plt.show()
        
        # plt.savefig(fileName+".png")
        
    # ---------------------------------------------------------------------------------
    
    # ================ EXP: EPOCAS =================

    if (int(sys.argv[4]) == 1 or sys.argv[1]== 'ALL' or int(sys.argv[1]) >=4 ):
        
        # DATA PREPARATION ------------------------
        # + medical experiments
        exp_file = "expblabla"
        columnToWatch = "layers"
        expectedInColumn = "10,8,6,4,2"
        
        
        df_resultados_med = pd.read_csv(f"results/{exp_file}.{fileType}",names=header_names)
        df_resultados_med = df_resultados_med[(df_resultados_med[columnToWatch]==expectedInColumn)]
        df_resultados_med = df_resultados_med.sort_values(by=['epoch'])
        
        # + electric experiments
        exp_file = "expblabla"
        columnToWatch = "layers"
        expectedInColumn = "10,8,6,4,2"
        
        
        df_resultados_elec = pd.read_csv(f"results/{exp_file}.{fileType}", names=header_names)
        df_resultados_elec = df_resultados_elec[(df_resultados_elec[columnToWatch]==expectedInColumn)]
        df_resultados_elec = df_resultados_elec.sort_values(by=['epoch'])
        
        #PLOT PREPARATION ------------------------
        title = "Experimento sobre epochs"
        fileName = "exp_epoch"
        l = "-"
        name1 = "Dataset Medico"
        name2 = "Dataset Electrico"
        xAxis = "epoch"
        yAxis = "error"


        fig,(ax1,ax2) = plt.subplots(1,2, figsize=(12,3))
        fig.suptitle(title.title())

        ax1.set_title(name1)
        ax1.set_ylabel("Error")
        ax1.set_xlabel("epoch")
        ax2.set_title(name2)
        ax2.set_ylabel("Error")
        ax2.set_xlabel("epoch")

        ax1.plot(df_resultados_med[xAxis],df_resultados_med[yAxis],l, c="blue", label="Med")
        ax2.plot(df_resultados_elec[xAxis],df_resultados_elec[yAxis],l, c="blue", label="Elec")

        ax1.legend()
        ax2.legend()

        plt.show()
        
        # plt.savefig(fileName+".png")
        
        
        
        
        
        
        
        
        
    # ---------------------------------------------------------------------------------