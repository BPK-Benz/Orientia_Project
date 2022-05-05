import pandas as pd
import os

path = os.getcwd()+'/../'

df = pd.read_csv(path+'Raw_Data/RawNonToxic_Scramble.csv').iloc[:,1:]
meanScram = df.groupby(['Gene_Symbol','Week','Plate']).mean().reset_index() # well data (mean)
meanGene = meanScram.groupby(['Gene_Symbol','Plate']).mean().reset_index().set_index('Gene_Symbol') # gene data (mean)

# Fold Change (mean)
new = pd.DataFrame()
for p in sorted(set(meanGene['Plate'])):
    temp = meanGene[meanGene['Plate'] == p].iloc[:,1:]
    divider = temp.loc['Scramble_siRNA',:]
    temp0 = temp/divider
    temp0 = temp0.add_prefix('FC_')
    add = pd.concat([meanGene[meanGene['Plate'] == p], temp0], axis=1)
    new = new.append(add) 

new.to_csv(path+'FoldChange/FC_NonToxic.csv')