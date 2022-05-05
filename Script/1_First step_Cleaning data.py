import pandas as pd
import numpy as np
import os
import glob
from collections import Counter


path = os.getcwd()+'/../'
temp1 = pd.read_csv(path+'Raw_Data/Modified_Original_genes.csv')

# Generate Toxic criteria = 1/3*Mean of scramble in each plate
Smean = pd.DataFrame(temp1[temp1['Gene_Symbol'] == 'Scramble_siRNA'].groupby(['Week','Plate']).mean()['Total_cells'])
Smean.columns = ['Mean']
Smean['OneThird(mean)'] = Smean['Mean']*1/3
Smean['Name'] = [_[0]+'_'+_[1] for _ in Smean.index]



# Evaluate toxic gene by above criteria and show the results with bar graph
temp1['Onethird(Smean)'] = [Smean.set_index('Name').loc[i+'_'+j,'OneThird(mean)'] for i,j in zip(temp1['Week'], temp1['Plate'])]
temp1['Toxic status'] = np.where( temp1['Total_cells'] < temp1['Onethird(Smean)'], 'Lower Mean', 'Equal/Upper Mean')


# Classify non-toxic genes and toxic genes
Nontoxic_df = pd.DataFrame() 
Toxic_df = pd.DataFrame()
for _ in set(temp1['Plate'].tolist()):
    temp1_plate = temp1[temp1['Plate'] == _]
    table = pd.pivot_table(data=temp1_plate,
               index=['Gene_Symbol'], columns=['Toxic status'], 
                aggfunc='count')['Total_cells'].fillna(0)
    gtable = table.drop(table[table.sum(axis=1)> 3].index.tolist())
    toxic = gtable[gtable['Lower Mean'] >= 2].index.tolist()
    
    gtable_drop = temp1_plate.set_index('Gene_Symbol').loc[gtable.index.tolist(),:].drop(toxic)
    Nontoxic_df = Nontoxic_df.append(gtable_drop)
    
    
    gtable_toxic = temp1_plate.set_index('Gene_Symbol').loc[gtable.index.tolist(),:].loc[toxic,:]
    Toxic_df = Toxic_df.append(gtable_toxic)
    

# Get 4 cleaning dataset 
Nontoxic_df['Inf/Uninf_cells'] = Nontoxic_df['Inf_cells']/Nontoxic_df['UnInf_cells'].round(4)
Nontoxic_df.reset_index().sort_values(by='Gene_Symbol').reset_index(drop=True).to_csv(path+'Raw_Data/RawNonToxic_gene.csv')
Toxic_df.reset_index().sort_values(by='Gene_Symbol').reset_index(drop=True).to_csv(path+'Raw_Data/RawToxic_gene.csv')
temp1[temp1['Gene_Symbol'] == 'Scramble_siRNA'].iloc[:,2:].reset_index(drop=True).to_csv(path+'Raw_Data/Cleansing_scramble.csv')
temp1[temp1['Gene_Symbol'] == 'Scramble_siRNA'].append(Nontoxic_df.reset_index(drop=True)).iloc[:,2:].reset_index(drop=True).to_csv(path+'Raw_data/All_genes.csv')