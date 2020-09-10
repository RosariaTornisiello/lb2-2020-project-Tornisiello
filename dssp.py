#!/usr/bin/python3
import sys

def get_ss(dsspfile):
    file=open(dsspfile)
    i=0
    seq=''
    s_structure=''
    dssp=[]
    chains=[]
    for line in file:
        if line.find('  #  RESIDUE')>-1:            #-1 means not found
            i=1
            continue
        if i==0 or len(line)<115:continue
        aa=line[13]
        ss=line[16]
        if aa!='!':
             seq+=aa
             if ss=='T' or ss=='S' or ss==' ': s_structure+='C'
             elif ss=='H' or ss=='G' or ss=='I' : s_structure+='H'
             elif ss=='B' or ss=='E': s_structure+='E'
             else : print('error', ss, 'not found')
        elif aa=='!':
            dssp.append(seq)
            dssp.append(s_structure)
            seq=''
            s_structure=''
    dssp.append(seq)
    dssp.append(s_structure)
    return dssp


# if you apply this program to protein data bank data sets containing oligomers,
# solvent exposure is for the entire assembly, not for the monomer
def get_ASA(dsspfile):          #accessible surface area
    file=open(dsspfile)
    i=0
    for line in file:
        if line.find('  #  RESIDUE')>-1:            #-1 means not found
            i=1
            continue
        if i==0 or len(line)<115:continue
        acc=float(line[35:38])
        print(acc)


if __name__=='__main__':
    dsspfile=sys.argv[1]
    dssp=get_ss(dsspfile)
    print(dssp)
    # for i in range(0,len(dssp),2):
    #     print('Sequence:','\n',dssp[i],'\n','Secondary structure:','\n',dssp[i+1])
    #asa=get_ASA(dsspfile)
