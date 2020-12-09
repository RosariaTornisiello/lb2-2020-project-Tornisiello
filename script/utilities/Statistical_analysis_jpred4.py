#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
jpred = pd.read_csv("jpred4.tsv", sep="\t")
jpred


# In[3]:


#1
import seaborn as sns
sns.boxplot(y="SCOPClass", x="Length", data=jpred)


# In[4]:


#2
import os
E = 0
H = 0
C = 0
for filename in os.listdir("dssp"):
    file = open('dssp/' + filename, "r")
    for line in file:
        if not line.startswith('>'):
            for letter in line:
                if letter == 'E':
                    E += 1
                if letter == 'H':
                    H += 1
                if letter == '-':
                    C += 1
L = []
L.append(E)
L.append(H)
L.append(C)
print(L)
        


# In[9]:


ss = pd.DataFrame({'ss': L},
                  index=['E', 'H', 'C'])
ss_composition = ss.plot.pie(y='ss', figsize=(5, 5), autopct = '%1.1f%%')
fig = ss_composition.get_figure()
fig.savefig("ss_composition.png")


# In[11]:


#3
aa = {}
for filename in os.listdir("fasta"):
    file = open('fasta/' + filename, "r")
    for line in file:
        line = line.rstrip()
        if not line.startswith('>'):
            for letter in line:
                if letter not in aa:
                    aa[letter] = 1
                else:
                    aa[letter] += 1
aa.pop('X')
print(aa)
tot = sum(aa.values())
print(tot)


# In[12]:


frequencies = {}
for key in aa:
    frequencies[key] = ((aa.get(key))/tot)*100
print(frequencies)


# In[14]:


aa_df = pd.DataFrame.from_dict(frequencies, orient='index', columns=['Tot'])
barplot = sns.barplot(x=aa_df.index, y=aa_df['Tot'], data=aa_df)
barplot = barplot.get_figure()
barplot.savefig('./plots_training/aa_composition.png')


# In[15]:


E = {}
H = {}
C = {}
for filefasta in os.listdir("fasta"):
    fileF = open('fasta/' + filefasta, "r")
    ID = filefasta[0:-6]
    fileD = open('dssp/' + ID + ".dssp", "r")
    for line in zip(fileF, fileD):
        if not line[0].startswith('>'):
            for x in range(len(line[0])):
                if line[1][x] == 'E':
                    E[line[0][x]] = E.get(line[0][x], 0) + 1                 
                elif line[1][x] == 'H':
                    H[line[0][x]] = H.get(line[0][x], 0) + 1
                elif line[1][x] == '-':
                    C[line[0][x]] = C.get(line[0][x], 0) + 1
print(H)               
print(E)
C.pop('X')
print(C)


# In[16]:


tot_Helix = sum(H.values())
tot_Coil = sum(C.values())
tot_Beta = sum(E.values())
freq_Helix = {}
for key in H:
    freq_Helix[key] = ((H.get(key))/tot_Helix)*100
freq_Coil = {}
for key in C:
    freq_Coil[key] = ((C.get(key))/tot_Coil)*100
freq_Beta = {}
for key in E:
    freq_Beta[key] = ((E.get(key))/tot_Beta)*100
    
Helix = pd.DataFrame.from_dict(freq_Helix, columns=['Helix'], orient='index')
Coil = pd.DataFrame.from_dict(freq_Coil, columns=['Coil'], orient='index')
Beta = pd.DataFrame.from_dict(freq_Beta, columns=['Beta'], orient='index')
ss_aa_tot = pd.concat([Beta, Helix, Coil, aa_df], axis=1)
ss_aa_tot


# In[17]:


ss_aa = pd.melt(ss_aa_tot, value_vars=['Beta', 'Helix', 'Coil', 'Tot'], var_name='ss', ignore_index=False)
ss_aa.reset_index(inplace=True)
ss_aa


# In[19]:


import matplotlib.pyplot as plt
sns.set()
plt.figure(figsize=(15, 4))
barplot = sns.barplot(x="index", y="value", hue="ss", data=ss_aa)
barplot = barplot.get_figure()
barplot.savefig('./plots_training/aa_composition_per_ss.png')


# In[94]:


#4a
sns.countplot(x="Suprekingdom", data=jpred)


# In[20]:


#alternativa al 4a
pie = jpred.Suprekingdom.value_counts().plot.pie()
pie = pie.get_figure()
pie.savefig('./plots_training/Superkingdom.png')


# In[6]:


#4b
import matplotlib as plt

count = jpred['TaxaName'].value_counts()
count_df = count.rename_axis('TaxaName').to_frame('counts')
count_df_10 = count_df[:10].copy()
others = pd.DataFrame(data = {'counts' : [count_df['counts'][10:].sum()]}, index = ['Others'])
tot_df = pd.concat([others,count_df_10])
tot_df['TaxaName'] = tot_df.index

tot_df.plot(kind = 'pie', y = 'counts',subplots=True, labels = tot_df['TaxaName'],figsize=(6,6))


# In[7]:


#altrnativa 4b
sns.barplot(y=tot_df['TaxaName'], x =tot_df['counts'],data=tot_df)


# In[21]:


#5
SCOP = sns.countplot(y="SCOPClass", data=jpred)
SCOP = SCOP.get_figure()
SCOP.savefig('./plots_training/SCOPclass.png')


# In[24]:


pie = jpred.SCOPClass.value_counts().plot.pie()
pie = pie.get_figure()
pie.savefig('./plots_training/SCOPClass2.png')


# In[ ]:




