import numpy as np
import os
import pandas as pd 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="file containing the list of input IDs")
parser.add_argument("--data", help="directory containing the input profiles including the ss seq")
parser.add_argument("--window", help="sliding window size")
parser.add_argument("--output", help="output file (the model)")
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

def generate_model(w):

    """ initializes a doctionary with secondary structures 
    as keys and x*20 np 0 arrays as values """

    sec_str = ['H', 'E', '-', 'R']
    model = {}
    for s in sec_str:
        model[s] = np.zeros((w, 20), dtype=int) 
    return(model)

#train model
def train_model(w, profile, model):

    """ trains the model by using a sliding window,
    summing it to the corresponding np array in the 
    model """

    start = 0
    end = (start + w)-1
    for line in profile:
        window = profile[start:(end+1)]
        start += 1
        end += 1
        if len(window) == w:
            structure = ''.join(window[[w//2], [21]])
            model[structure] = np.add(model[structure], window[:, 0:20])
            model['R'] = np.add(model['R'], window[:, 0:20])
    return(model)

#normalize
def normalize_model(w, model):

    """ normalizes the value in each posizion of the model
    by diving it by the sum of the values in the corresponding
    row of the R np array """

    for i in range(w):
        l = ['H', 'E', '-', 'R']
        for s in l:
            model[s][i] = model[s][i]/np.sum(model['R'][i])
    return(model)

def final_model(w, model):

    """ converts the model into a multiindex dataframe """
    
    H = model['H']
    E = model['E']
    C = model['-']
    R = model['R']
    array = np.concatenate((H, E, C, R), axis=0) 
    aa = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    labels = []
    for i in range((-w//2)+1, (w//2)+1 ,1):
        labels.append(i)
    index = pd.MultiIndex.from_product([model.keys(), labels])
    model_df = pd.DataFrame(array, columns=aa, index=index)
    return(model_df)

w = int(args.window)
IDlist = [line.rstrip('\n') for line in open(args.input, 'r')]
model = generate_model(w)
for filename in os.listdir(args.data):
    ID = filename[0:-8]
    if ID in IDlist:
        pro_file = open(args.data + '/' + ID + '.profile')
        profile = pd.read_csv(pro_file, sep='\t')
        profile_pad = padding(w, profile)
        profile = profile_pad.to_numpy()
        trained_model = train_model(w, profile, model)
normalized_model = normalize_model(w, trained_model)
normalized_model_df = final_model(w, normalized_model) 
output = open(args.output, 'w')
normalized_model_df.to_csv(output, sep='\t')

