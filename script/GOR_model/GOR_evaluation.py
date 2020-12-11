from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, classification_report
import os
import pandas as pd
from scipy.stats import sem
import statistics

def my_MCC(y_true, y_pred, ss):
    MCC = dict.fromkeys(ss, 0)
    for structure in ss:
        y_true_temp = [structure if i == structure else 'X' for i in y_true]
        y_pred_temp = [structure if i == structure else 'X' for i in y_pred]
        MCC[structure] = matthews_corrcoef(y_true_temp, y_pred_temp)
    return(MCC)

def generate_df(ss):
    columns = ['H', '-', 'E'] 
    iterables = [['split0', 'split1', 'split2', 'split3', 'split4', 'blind'], ['MCC', 'precision', 'recall']]
    my_index = pd.MultiIndex.from_product(iterables, names=['data', 'score'])
    df = pd.DataFrame(index= my_index, columns=columns )
    return(df)

ss = ['H', '-', 'E']
df = generate_df(ss)

#fill df with cv values
Q3_list_cv = []
for i in range(5):
    y_true = list(open('/home/rosaria/Desktop/LAB2/lb2-2020-project-Tornisiello/data/GOR_results/split'+ str(i) + '/' + 'y_true_' + str(i) + '.txt').read())
    y_pred = list(open('/home/rosaria/Desktop/LAB2/lb2-2020-project-Tornisiello/data/GOR_results/split'+ str(i) + '/' + 'y_pred_' + str(i) + '.txt').read())
    MCC = my_MCC(y_true, y_pred, ss)
    report = classification_report(y_true, y_pred, labels=ss, output_dict=True)
    Q3 = accuracy_score(y_true, y_pred)
    Q3_list_cv.append(Q3)
    for s in ss:
         df.loc['split'+str(i), 'MCC'][s] = MCC[s]
         df.loc['split'+str(i), 'precision'][s] = report[s]['precision']
         df.loc['split'+str(i), 'recall'][s] = report[s]['recall']
Q3_split = statistics.mean(Q3_list_cv)


#fill df with blind valuesx
y_true = list(open('/home/rosaria/Desktop/LAB2/lb2-2020-project-Tornisiello/data/GOR_results/blind_test_pred/y_true_blind.txt').read())
y_pred = list(open('/home/rosaria/Desktop/LAB2/lb2-2020-project-Tornisiello/data/GOR_results/blind_test_pred/y_pred_blind.txt').read())
MCC = my_MCC(y_true, y_pred, ss) # 2 dictionaries
report = classification_report(y_true, y_pred, labels=ss, output_dict=True) #dictionary of dictionaries
Q3_blind = accuracy_score(y_true, y_pred)
for s in ss:
    df.loc['blind', 'MCC'][s] = MCC[s]
    df.loc['blind', 'precision'][s] = report[s]['precision']
    df.loc['blind', 'recall'][s] = report[s]['recall']
print(df)
print("Q3 cv:", Q3_split)
print("Q3 blind:", Q3_blind)

def standard_error_per_MCC(df, ss):
    """ computes st err for each score for the given ss as input""" 
    MCC_list = []
    for i in range(5):
        MCC_list.append(df.loc['split'+str(i), 'MCC'][ss])
    mean = statistics.mean(MCC_list)
    st_err_MCC = sem(MCC_list)
    return('for', ss, '-->','MCC:', mean, '+/-', st_err_MCC)


#st_err_MCC, st_err_precision, st_err_recall = standard_error_per_score(df, 'H')
for s in ss:
    print(standard_error_per_MCC(df, s))


