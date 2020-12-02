import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="directory containing the input profiles including the ss seq")
parser.add_argument("--window", help="sliding window size")
parser.add_argument("--output", help="output file (the model)")
args = parser.parse_args()

def df_labels(w):
    residue_ss = []
    ss = ['H', 'E', '-', 'R']
    for letter in ss:
        for i in range((-w//2)+1, (w//2)+1 ,1):
            residue_ss.append(str(i))
            residue_ss.append(letter)
            residue_ss.append('X')
    residue_ss = ''.join(residue_ss)
    residue_ss = residue_ss.split('X')
    residue_ss = residue_ss[:-1]
    return(residue_ss)

def generate_df(w, labels_list):
    matrix = np.zeros((w*4, 20))
    aa = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    df = pd.DataFrame(matrix, columns=aa)
    df.index = labels_list
    return(df)

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

def generate_model(w, profile, df):
    start = 0
    end = (0 + w)-1
    for _ in profile.iterrows():
        window = profile.loc[start:end]
        start += 1
        end += 1
        if len(window.index) == w:
            middle = window.iloc[w//2]
            ss = middle['ss']
            labels = window_labels(w, ss)
            window.index = labels
            df = df.add(window.iloc[0:w, 0:20], fill_value=0)
            R_labels = window_labels(w, 'R')
            window.index = R_labels
            df = df.add(window.iloc[0:w, 0:20], fill_value=0)
    return(df)

def normalize(w, df):
    aa = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    R_labels = window_labels(w, 'R')
    R = pd.DataFrame(columns=aa, index=R_labels)
    for label in R_labels:
        R.loc[label]=df.loc[label]
    sum = R.sum(axis=1)
    ss = ['H', 'E', '-', 'R']
    for line in sum.index:
        for s in ss:
            index = str(line[:-1]) + s
            df.loc[index] = df.loc[index].div(sum[line])
    return(df)

w = int(args.window)
labels_list = df_labels(w)
df = generate_df(w, labels_list)
for filename in os.listdir(args.data):
    ID = filename[0:-8]
    pro_file = open(args.data + '/' + ID + '.profile')
    profile = pd.read_csv(pro_file, sep='\t')
    profile_pad = padding(w, profile)
    df = generate_model(w, profile_pad, df)
normalized_model = normalize(w, df)
output = open(args.output, 'w')
normalized_model.to_csv(output, sep='\t')
