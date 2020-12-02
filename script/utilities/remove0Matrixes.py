import pandas as pd
import os

c = 0
for filename in os.listdir('/home/rosaria/Desktop/LAB2/LAB2_project/sequence_profile/blind_profile/'):
    id = filename[0:-8]
    file = '/home/rosaria/Desktop/LAB2/LAB2_project/sequence_profile/blind_profile/'+id+'.profile'
    df = pd.read_csv(file, sep='\t')
    bool = df.loc[:, 'A':'V'].any(axis=None)
    if bool == False:
        c += 1
        os.remove(file)
print(c, 'file removed!')
