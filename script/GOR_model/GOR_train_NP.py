import numpy as np
import os
import pandas as pd 
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="file containing the list of input IDs")
parser.add_argument("--data", help="directory containing the input profiles including the ss seq")
parser.add_argument("--window", help="sliding window size")
parser.add_argument("--output", help="output file (the information model)")
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
def train_model(w, profile, model, secondary):

    """ trains the model by using a sliding window,
    summing it to the corresponding np array in the 
    model """

    start = 0
    end = (start + w)-1
    for _ in profile:
        window = profile[start:(end+1)]
        start += 1
        end += 1
        if len(window) == w:
            structure = ''.join(window[[w//2], [21]])
            secondary.loc[['probability'], [str(structure)]] = secondary.loc[['probability'], [str(structure)]] + 1
            model[structure] = np.add(model[structure], window[:, 0:20])
            model['R'] = np.add(model['R'], window[:, 0:20])
    #secondary = secondary.div(secondary.sum())
    return(model, secondary)

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

def model_todf(w, model):

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


def information_model(normalized_model, secondary):

    """ converts the normalized model into information 
    matrix """

    info_model = normalized_model.copy(deep=True)
    for index in normalized_model.index:
        for column in normalized_model.columns:
            PRS = normalized_model.loc[index, column]
            PSS = secondary.loc['probability'].at[index[0]]
            PR = normalized_model.loc[('R', index[1]), column]
            info_model.loc[index, column] = math.log(PRS/(PSS*PR),2)
    return(info_model)

w = int(args.window)
IDlist = [line.rstrip('\n') for line in open(args.input, 'r')]
model = generate_model(w)
sec_str = ['H', 'E', '-']
secondary = pd.DataFrame(columns=sec_str, index=['probability'])
for filename in os.listdir(args.data):
    ID = filename[0:-8]
    if ID in IDlist:
        pro_file = open(args.data + '/' + ID + '.profile')
        profile = pd.read_csv(pro_file, sep='\t')
        profile = padding(w, profile).to_numpy()
        model, secondary = train_model(w, profile, model, secondary)
print(secondary)
# model = normalize_model(w, model)
# model = model_todf(w, model)
# model = information_model(model, secondary) 
# output = open(args.output, 'w')
# model.to_csv(output, sep='\t')

