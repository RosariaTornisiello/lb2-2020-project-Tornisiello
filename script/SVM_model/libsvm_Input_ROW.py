import numpy as np
import pandas as pd
import os 
import csv

def padding(w, profile):
    aa = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    matrix = np.zeros((w//2, 20))
    df = pd.DataFrame(matrix, columns=aa)
    profile = pd.concat([df, profile])
    profile = pd.concat([profile, df])
    profile.reset_index(inplace=True, drop=True)
    return(profile)

def s_number(structure):
    if structure == 'H':
        num = 1
    elif structure == 'E':
        num = 2
    else: num = 3
    return(num)

def generate_input(pad_profile, w):
    sequence = []
    start = 0
    end = (start + w)-1
    for line in profile:
        window = profile[start:(end+1)]
        start += 1
        end += 1
        if len(window) == w:
            structure = window[w//2, 21]
            num = s_number(structure) 
            window = np.delete(window, [0, 21], 1)
            window = window.flatten()
            line = window.tolist()
            line.insert(0, num)
            for i in range(1, 341):
                line[i] = str(i) + ":" + str(line[i])
            sequence.append(line)
    return(sequence)


#for filename in os.listdir('/home/rosaria/Desktop/LAB2/LAB2_project/cv/'):
lineList = [line.rstrip('\n') for line in open('/home/rosaria/Desktop/LAB2/LAB2_project/cv/set0', 'r')]
split = []
for ID in lineList:
    path = '/home/rosaria/Desktop/LAB2/LAB2_project/sequence_profile/training_profile/' + ID + '.profile'
    if os.path.isfile(path) == True: 
        pro_file = open(path, 'r')
        profile = pd.read_csv(pro_file, sep='\t')
        profile_pad = padding(17, profile)
        profile = profile_pad.to_numpy()
        sequence = generate_input(profile, 17)
        split = split + sequence
with open("./Set00.csv", "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(split)
