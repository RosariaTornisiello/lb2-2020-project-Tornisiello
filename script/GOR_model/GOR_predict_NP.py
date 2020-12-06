import os
import pandas as pd
import numpy as np
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="file containing the list of input IDs")
parser.add_argument("--data", help="directory containing the input test profiles")
parser.add_argument("--model", help="information model")
parser.add_argument("--output", help="y_pred output file")
parser.add_argument("--window", help="sliding window size")
args = parser.parse_args()

def padding(w, profile):

    """ pads the profile given in input: 
    adds a 0 matrix of lenght w/2 at the beginning 
    and at the end of the profile """

    aa = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    matrix = np.zeros((w//2, 20))
    df = pd.DataFrame(matrix, columns=aa)
    profile = pd.concat([df, profile])
    profile = pd.concat([profile, df])
    profile.reset_index(inplace=True, drop=True)
    return(profile)

def ss_probability(w, window, info_model):

    """ generates a dictionary with ss as keys and theire 
    probability for each window"""

    ss = ['H','E', '-']
    prob = dict.fromkeys(ss)
    for s in ss:
        ss_window = np.multiply(window, info_model.loc[[s]])
        prob[s] = ss_window.values.sum()
    return(prob)


def predict_ss(w, profile, info_model, y_pred):

    """ predicts the secondary structure for each residue
    of a protein given its profile"""
    
    start = 0
    end = (0 + w)-1
    for _ in profile:
        window = profile[start:(end+1), 0:20]
        start += 1
        end += 1
        if len(window) == w:
            prob = ss_probability(w, window, info_model)
            if all(x==0 for x in prob.values()) == True:
                y_pred += '-'
            else:
                prediction = max(prob, key=prob.get)                      
                y_pred += prediction
    return(y_pred)
    

w = int(args.window)
model = args.model
model = pd.read_csv(model, sep='\t', index_col=[0, 1])
y_pred = ''
IDlist = [line.rstrip('\n') for line in open(args.input, 'r')]
for ID in IDlist:
    filename = ID+'.profile'
    if filename in os.listdir(args.data):
        pro_file = args.data + '/' + filename
        profile = pd.read_csv(pro_file, sep='\t', index_col=0)
        profile = padding(17, profile)
        profile = profile.to_numpy()
        y_pred = predict_ss(w, profile, model, y_pred)
output = open(args.output, 'w')
print(y_pred, file = output)

