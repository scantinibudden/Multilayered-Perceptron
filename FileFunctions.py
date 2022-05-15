import numpy as np
import glob

file_name_template = "results/results_#exp_name#_#run_number#.txt"

def store(exp_name, df):
    file_name = file_name_template.replace("#exp_name#", exp_name)
    run_number = getExpRunNumber(file_name)
    file_name = file_name.replace("#run_number#", run_number)
    with open(file_name, 'wb') as file:
        np.savetxt(file, df, delimiter = ",", fmt='%s')

def getExpRunNumber(file_name):
    file_name_pattern = file_name.replace("#run_number#", "*")
    file_list = sorted(glob.glob(file_name_pattern)) #results/exp_1/results_*.csv
    if (len(file_list) == 0):
        return str(0)
    file_list = [int(x.split('_')[-1][:-4]) for x in file_list]
    return str(sorted(file_list)[-1] + 1)

def stringify(values):
    res = []
    for elem in values:
        if isinstance(elem, list):
            res.append(arrToString(elem))
        else:
            res.append(str(np.around(elem, 3)))
    return res

def arrToString(arr):
    res = ""
    for elem in arr:
        res += str(elem) + "-"
    res = res[:-1]
    return res

def stringToArr(str):
    return str.split("-")