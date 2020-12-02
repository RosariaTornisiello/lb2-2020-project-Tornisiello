import os
import pandas as pd

for filename in os.listdir('/home/rosaria/Desktop/LAB2/LAB2_project/sequence_profile/blind_pssm/'):
    id = filename[0:-5]
    file = open('/home/rosaria/Desktop/LAB2/LAB2_project/sequence_profile/blind_pssm/'+id+'.pssm', 'r')
    dssp = open('/home/rosaria/Desktop/LAB2/LAB2_project/blind_dssp_chains/'+id+'.dssp')
    profile = []
    for line in file:
        line.rstrip()
        l = line.split()
        profile.append(l)
    ss = []
    for line in dssp:
        if line.startswith('>') == False:
            ss[:0] = line
            ss.pop()
    x = ''
    profile[2].insert(0, x)
    profile[2].insert(0, x)
    df = pd.DataFrame(profile)
    df = df.drop([0,1])
    df = df.drop(df.tail(6).index)
    df = df.drop(columns=[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,42,43])
    df.reset_index(drop=True, inplace=True)
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df.loc[:, 'A':'V'] = df.loc[:, 'A':'V'].astype(float)
    df.loc[:, 'A':'V'] = df.loc[:, 'A':'V'].div(100)
    df.rename(columns={df.columns[0]: 'seq'}, inplace=True)
    df['ss'] = ss
    if df.shape[0]>=49:
        df.to_csv('/home/rosaria/Desktop/LAB2/LAB2_project/sequence_profile/blind_profile/'+id+'.profile', sep='\t', index=False)
