import os

for filename in os.listdir('/home/rosaria/Desktop/LAB2/LAB2_project/150_chains/'):
    chain = filename[5]
    file_dssp = open('/home/rosaria/Desktop/LAB2/LAB2_project/dssp_outputs/'+filename+'.dssp', 'r')
    aa_sequence = ''
    s_structure = ''
    c=0
    for line in file_dssp:
        if line.find('  #  RESIDUE') > -1:
            c=1
            continue
        if c==0 or len(line)<115:continue
        aa = line[13]
        ss = line[16]
        if aa != '!':
            if aa.islower():
                aa_sequence += 'C'
            else: aa_sequence += aa
            if ss=='T' or ss=='S' or ss==' ': s_structure+='-'
            elif ss=='H' or ss=='G' or ss=='I' : s_structure+='H'
            elif ss=='B' or ss=='E': s_structure+='E'
            else : print('error', ss, 'not found')
        elif aa == '!':
            aa_sequence += 'X'
            s_structure += '-'
    entry = filename[0:6]
    file_fasta = open('/home/rosaria/Desktop/LAB2/LAB2_project/blind_fasta_chains/'+entry+'.fasta', 'w')
    print('>' + entry + '\n' + aa_sequence, file = file_fasta)
    f_dssp = open('/home/rosaria/Desktop/LAB2/LAB2_project/blind_dssp_chains/'+entry+'.dssp', 'w')
    print('>' + entry + '\n' + s_structure, file = f_dssp)
