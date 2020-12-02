from sklearn.metrics import matthews_corrcoef
import statistics
 


def myMCC(y_true, y_pred):
    ss = [1, 2, 3]
    MCCs = []
    for structure in ss:
        y_true_temp = list()
        y_pred_temp = list()
        y_true_temp = [structure if i == structure else 'X' for i in y_true]
        y_pred_temp = [structure if i == structure else 'X' for i in y_pred]
        MCC = matthews_corrcoef(y_true_temp, y_pred_temp)
        MCCs.append(MCC)
    return(statistics.mean(MCCs))

y_true = [1, 2, 3, 3, 2, 1, 2, 3, 2, 1, 2, 2, 2, 3, 2, 1]
y_pred = [1, 2, 2, 3, 3, 1, 2, 3, 2, 1, 2, 2, 1, 3, 2, 2]
print(myMCC(y_true, y_pred))
