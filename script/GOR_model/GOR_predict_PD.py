import os
import pandas as pd
import numpy as np
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="directory containing the input test profiles")
parser.add_argument("--model", help="normalized GOR model")
parser.add_argument("--output", help="directory for output files")
parser.add_argument("--window", help="sliding window size")
args = parser.parse_args()

def padding(w, profile):
    aa = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    matrix = np.zeros((w//2, 20))
    df1 = pd.DataFrame(matrix, columns=aa)
    df2 = pd.DataFrame(matrix, columns=aa)
    profile = pd.concat([df1, profile])
    profile = pd.concat([profile, df2])
    profile.reset_index(inplace=True, drop=True)
    return(profile)

def window_labels(w, ss):
    labels = []
    for i in range((-w//2)+1, (w//2)+1 ,1):
        labels.append(str(i))
        labels.append(str(ss))
        labels.append('X')
    labels = ''.join(labels)
    labels = labels.split('X')
    labels = labels[:-1]
    return(labels)

def ss_vector(model):
    ss = ['H', 'E', '-', 'R']
    secondary = pd.DataFrame(columns=ss, index=['probability'])
    for letter in ss:
        secondary.loc['probability'].at[letter] = model.loc[str(0)+letter].sum()
    return(secondary)


def information_model(model, secondary):
    info_model = model.copy(deep=True)
    for index in model.index:
        R_label = str(index[:-1]+'R')
        for column in model.columns:
            PRS = model.loc[index].at[column]
            PSS = secondary.loc['probability'].at[index[-1]]
            PR = model.loc[R_label].at[column]
            info_model.loc[index].at[column] = math.log(PRS/(PSS*PR),2)
    return(info_model)

#prediction

def max_probability(w, window, info_model):
    ss = ['H','E', '-']
    prob = dict.fromkeys(ss)
    for s in ss:
        labels = window_labels(w, s)
        window.index = labels
        ss_window = window.mul(info_model.loc[labels])
        prob[s] = ss_window.sum().sum()
    return(prob)



def predict_ss(w, pad_profile, info_model):
    start = 0
    end = (0 + w)-1
    predicted_ss = ''
    for lines in pad_profile.iterrows():
        window = pad_profile.loc[start:end]
        start += 1
        end += 1
        if len(window.index) == w:
            prob = max_probability(w, window, info_model)
            if any(prob.values()) == False:
                predicted_ss += '-'
            else:
                prediction = max(prob, key=prob.get)                      
                predicted_ss += prediction
    return(predicted_ss)


w = int(args.window)
file = args.model
model = pd.read_csv(file, sep='\t', index_col=0)
secondary = ss_vector(model)
info_model = information_model(model, secondary)

for filename in os.listdir(args.data):
    ID = filename[0:-8]
    pro_file = args.data + '/' + ID + '.profile'
    profile = pd.read_csv(pro_file, sep='\t', index_col=0)
    pad_profile = padding(17, profile)
    output = open(args.output + '/' + ID + '.predicted', 'w')
    print('>' + ID + '\n' + predict_ss(w, pad_profile, info_model), file = output)