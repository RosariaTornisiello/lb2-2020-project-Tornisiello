# takes IDs as input
# apre profilo
# prende ss
# trasforma in stringa

import numpy as np
import os
import pandas as pd 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="file containing the list of input IDs")
parser.add_argument("--data", help="directory containing the input profiles including the ss seq")
parser.add_argument("--output", help="directory to which redirect output file")
args = parser.parse_args()

y_true = ''
IDlist = [line.rstrip('\n') for line in open(args.input, 'r')]
for ID in IDlist:
    if ID+'.profile' in os.listdir(args.data):
        profile = open(args.data + ID + '.profile')
        profile = pd.read_csv(profile, sep='\t')
        ss = profile['ss'].astype('string')
        ss = ss.str.cat()
        y_true += ss
output = open(args.output, 'w')
print(y_true, file=output)
