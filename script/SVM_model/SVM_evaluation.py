from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report, multilabel_confusion_matrix
import os
import pandas as pd
from scipy.stats import sem
import joblib
import numpy as np
import pandas as pd

def my_MCC(y_true, y_pred, ss):
    MCC = dict.fromkeys(ss, 0)
    for structure in ss:
        y_true_temp = [structure if i == structure else 'X' for i in y_true]
        y_pred_temp = [structure if i == structure else 'X' for i in y_pred]
        MCC[structure] = matthews_corrcoef(y_true_temp, y_pred_temp)
    return(MCC)

def convert_class_to_num(L):

    """ converts secondary structure in class number
    H = 1, E = 2, C = 3 """

    for i in range(len(L)):
        if L[i] == 'H':
            L[i]= 1
        elif L[i] == 'E':
            L[i]= 2
        else: L[i]= 3
    return(L)


y_pred_blind = joblib.load('/home/rosaria/Desktop/LAB2/lb2-2020-project-Tornisiello/data/SVM_results/y_pred_blind_SVM.joblib')
y_pred_blind = y_pred_blind.tolist()
y_pred_blind = [int(i) for i in y_pred_blind]

y_true_blind = list(open('/home/rosaria/Desktop/LAB2/lb2-2020-project-Tornisiello/data/SVM_results/y_true_blind.txt').read())
y_true_blind.pop()
y_true_blind = convert_class_to_num(y_true_blind)


ss = [1, 2, 3]
MCC = my_MCC(y_true_blind, y_pred_blind, ss)
report = classification_report(y_true_blind, y_pred_blind, labels=ss, output_dict=True)
Q3 = accuracy_score(y_true_blind, y_pred_blind)
index = ['MCC', 'precision', 'recall']
df = pd.DataFrame(index=index, columns=ss)
for s in ss:
    df.loc['MCC'][s] = MCC[s]
    df.loc['precision'][s] = report[str(s)]['precision']
    df.loc['recall'][s] = report[str(s)]['recall']
print(df)
print("Q3 =", Q3)