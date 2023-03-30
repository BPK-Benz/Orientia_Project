import pandas as pd
import numpy as np
import os 
import glob
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew, mode
sns.set_style("whitegrid")


# set dictionary betweeen Plate order and Batch order
map_dict = {
 'Plate_P01': 'Batch_01',
 'Plate_P02': 'Batch_01',
 'Plate_P03': 'Batch_01',
 'Plate_P04': 'Batch_01',
 'Plate_P05': 'Batch_02',
 'Plate_P06': 'Batch_02',
 'Plate_P07': 'Batch_02',
 'Plate_P08': 'Batch_02',
 'Plate_P09': 'Batch_03',
 'Plate_P10': 'Batch_03',
 'Plate_P11': 'Batch_03',
 'Plate_P12': 'Batch_03',
 'Plate_P13': 'Batch_03',
 'Plate_P14': 'Batch_03',
 'Plate_P15': 'Batch_04',
 'Plate_P16': 'Batch_04',
 'Plate_P17': 'Batch_04',
 'Plate_P18': 'Batch_04',
 'Plate_P19': 'Batch_04',
 'Plate_P20': 'Batch_04',
 'Plate_P21': 'Batch_04',
 'Plate_P22': 'Batch_05',
 'Plate_P23': 'Batch_05',
 'Plate_P24': 'Batch_05',
 'Plate_P25': 'Batch_05',
 'Plate_P26': 'Batch_05',
 'Plate_P27': 'Batch_05',
 'Plate_P28': 'Batch_05',
 'Plate_P29': 'Batch_05',
 'Plate_P30': 'Batch_06',
 'Plate_P31': 'Batch_06',
 'Plate_P32': 'Batch_06',
 'Plate_P33': 'Batch_06',
 'Plate_P34': 'Batch_06',
 'Plate_P35': 'Batch_06',
 'Plate_P36': 'Batch_06',
 'Plate_P37': 'Batch_06',
 'Plate_P38': 'Batch_07',
 'Plate_P39': 'Batch_07',
 'Plate_P40': 'Batch_07',
 'Plate_P41': 'Batch_07',
 'Plate_P42': 'Batch_07',
 'Plate_P43': 'Batch_07',
 'Plate_P44': 'Batch_07',
 'Plate_P45': 'Batch_07',
 'Plate_P46': 'Batch_08',
 'Plate_P47': 'Batch_08',
 'Plate_P48': 'Batch_08',
 'Plate_P49': 'Batch_08',
 'Plate_P50': 'Batch_08',
 'Plate_P51': 'Batch_08',
 'Plate_P52': 'Batch_08',
 'Plate_P53': 'Batch_08',
 'Plate_P54': 'Batch_09',
 'Plate_P55': 'Batch_09',
 'Plate_P56': 'Batch_09',
 'Plate_P57': 'Batch_09',
 'Plate_P58': 'Batch_09',
 'Plate_P59': 'Batch_09',
 'Plate_P60': 'Batch_09',
 'Plate_P61': 'Batch_09'}

 # Set color for each Batch
colors ={'Batch_01':'#f967c6',
         'Batch_02':'#b937f6',
         'Batch_03':'#376af6',
         'Batch_04':'#37e3f6',
         'Batch_05':'#87f637',
         'Batch_06':'#ff7300',
         'Batch_07':'#f6b637',
         'Batch_08':'gray',
         'Batch_09':'#3d741b',
         'Batch_10':'black'}


# Define function
# 1. SSMD 
def ssmd(ss, gg):
    up = np.mean(gg)-np.mean(ss)
    down = np.power((np.power(np.std(gg),2) + np.power(np.std(ss),2)),0.5)
    r = "Block" if np.mean(gg)-np.mean(ss) < 0 else "Uptake"
    return np.round(up/down,4), r, np.mean(gg), np.std(gg), np.mean(ss), np.std(ss)

# 2. IQR
def get_IQR():
    q1 = ssmd_df["val"].quantile(0.25)
    q3 = ssmd_df["val"].quantile(0.75)
    iqr = (df["val"] > q1) & (df["val"] < q3)
    return val.loc[iqr]

# 3. Conservative technique
def conservative(dependent, process, interpret, output_folder, name):
    # Calculate SSMD between scramble and knockdown genes 
    ssmdvalue = []

    for p in sorted(set(all_df['Plate'])):
        print(p)
        # Cut the window in 2 parts
        temp = all_df[all_df['Plate'] == p]

        for ii, g in enumerate(set(temp['Gene_Symbol'].tolist())-set('Scramble_siRNA')):
            ss = temp[(temp['Gene_Symbol'] == 'Scramble_siRNA')][dependent].tolist()
            gg = temp[(temp['Gene_Symbol'] == g)][dependent].tolist()
            z,n,mg,sdg, ms, sds = ssmd(ss,gg)
            ssmdvalue = ssmdvalue + [[g,p,z,n,mg,sdg, ms, sds]]

    ssmd_df = pd.DataFrame(ssmdvalue, columns=['Gene_Symbol','Plate',process , interpret,'Mean_gene','STD_gene','Mean_scramble','STD_scramble']).set_index('Gene_Symbol')
    ssmd_df['Batch'] = [map_dict[f] for f in ssmd_df['Plate']]

    # Calculate statistical value
    qs = ssmd_df.groupby('Plate')[process].quantile([0.25,0.75])
    qs = qs.unstack().reset_index()
    qs.columns = ["Plate", "q1", "q3"]
    qs['IQR'] = qs['q3']-qs['q1']
    qs['Median'] = pd.DataFrame(ssmd_df.groupby('Plate')[process].median())[process].tolist()
    qs['Batch'] = [map_dict[f] for f in qs['Plate']]
    qs['Minimum'] = ssmd_df.groupby('Plate')[process].min().tolist()
    qs['Maximum'] = ssmd_df.groupby('Plate')[process].max().tolist()
    qs = qs.set_index('Plate')
    return ssmd_df,qs

# set root path
path = os.getcwd()+'/../'

# # Import data

all_df = pd.read_csv(path+'Raw_Data/All_genes.csv')

# 2. Calculated ssmd for all genes and keep ssmd and median +/- interqualtile range
# 2.1 Bacteria entry process
output_folder = path+"Conservative_method/"
ssmd_entry, qs_entry = conservative(dependent='Bac/Inf', process='ssmd_Entry', interpret='Interpret_Entry', output_folder=output_folder, name='Entry')
ssmd_entry.drop(index=('Scramble_siRNA')).to_csv(output_folder+'/ssmd_entry.csv')
qs_entry.to_csv(output_folder+'/qs_entry.csv')

# 2.2 Bacteria translocation process
ssmd_translocation, qs_translocation = conservative(dependent='Nuc/Inf', process='ssmd_Trans', interpret='Interpret_Trans', output_folder=output_folder, name='Translocation(Outlier)')
ssmd_translocation.drop(index=('Scramble_siRNA')).to_csv(output_folder+'/ssmd_translocation.csv')
qs_translocation.to_csv(output_folder+'/qs_translocation.csv')

# 3. Save all data and rearrange column name
col = ['Total_cells', 'Inf_cells', 'UnInf_cells',
       'Bac/Inf', 'Nuc/Inf', 'Cytomem/Inf', 'Infected/Total_cells',
       'Nuc/IntraBac', 'Inf/Uninf_cells']

gene = all_df[all_df['Gene_Symbol'] != 'Scramble_siRNA'].iloc[:,1:]
for _ in glob.glob(output_folder+'ssmd_*.csv')[:2]:
    print(_)
    temp = pd.read_csv(_).set_index('Gene_Symbol')
    genelist = temp.index.tolist()
    value = gene.groupby('Gene_Symbol').mean().loc[genelist, col]
    new_df = pd.concat([value,temp], axis=1)
    new_df[temp.columns[2]] = new_df[temp.columns[2]].replace({'Block':'Inhibit', 'Uptake':'Enhance'})
    new_df.to_csv(_)
    print('............Bang.......')
    

###### Load: translocation and adaptive for hit and non-hit genes
## df1 is dataframe of hit genes
## df2 is dataframe of non-hit genes from conservative technique
## df3 is dataframe of non-hit genes from adaptive technique
###################################################################

##### Bacteria Entry process ##########################################################################################
## 1. Load hit genes
entry = pd.read_csv(path+'Final_output/Final_Entry.csv', index_col=0)
hit = entry.set_index(['Gene','Plate','Interpret_Entry']).index.tolist()


########################### Conservative analysis for non-hit genes (levels > 1500) #############################

ssmd_entry = pd.read_csv(glob.glob(path+'Conservative_method/ssmd_entry_labelled_*.csv')[0], index_col=0)
ssmd_entry = ssmd_entry[ssmd_entry['Gene'].notna()]
ssmd_entry = ssmd_entry.rename(columns={'Gene_Symbol':'Gene', 'UnInf_cells':'Uninf_cells'})
ssmd_entry['Phenotype'] = ssmd_entry['Interpret_Entry'].replace({'Block':'Inhibit', 'Uptake':'Enhance'})
non_hit = ssmd_entry.set_index(['Gene','Plate','Interpret_Entry']).index.tolist()

gene = list(set(non_hit) - set(hit)) # find non-hit genes 

df1 = entry.set_index(['Gene','Plate','Interpret_Entry'])
df1['Hit'] = ['hit']*df1.shape[0]
df1 = df1.drop(columns=['ssmd_Entry.1','Batch.1','Entry_Level'])

df2 = ssmd_entry.set_index(['Gene','Plate','Interpret_Entry']).loc[gene,:]
df2['Hit'] = ['non-hit']*df2.shape[0]
df2['Process'] = ['Entry']*df2.shape[0]
df2['LevelShort'] = ['> 1500']*df2.shape[0]
df2['Level'] = ['Con_Level_>1500']*df2.shape[0]
df2 = df2.drop(columns=['Plate.1', 'Batch.1'])


# load scramble data 
scramble = pd.read_csv(path+'Raw_Data/Cleansing_scramble_withoutWeek.csv', index_col=0)
meanScramble = scramble.groupby('Plate').mean()
meanScramble['Infected/Total_cells'] = meanScramble['Inf_cells']/meanScramble['Total_cells']
meanScramble['Nuc/IntraBac'] = meanScramble['Nuc/Inf']/meanScramble['Bac/Inf']
meanScramble['Inf/Uninf_cells'] = meanScramble['Inf_cells']/meanScramble['UnInf_cells']

temp = df2.reset_index()
value = []
for g, _ in zip(temp['Gene'],temp['Plate']):
    value = value + [meanScramble.loc[_,:].values.tolist()]
Scol = ['Scramble_'+_+'_mean' for _ in meanScramble.columns]
df2 = pd.concat([df2.reset_index(),pd.DataFrame(value, columns=Scol)], axis=1).set_index(['Gene','Plate','Interpret_Entry'])


############### Adaptive analysis for non-hit genes (levels > 1500) #######################################################

adap_entry = pd.read_csv(glob.glob(path+'Adaptive_RegressionModel/*Entry_ssmd*label*')[0]).drop(columns=['Week_Order', 'Week_Order_1', 'Week_Order_2','Week_Order_3'])
adap_entry = adap_entry.rename(columns={'Gene_Symbol':'Gene', 'UnInf_cells':'Uninf_cells','Scramble_Entry (Model)':'Scramble_Bac/Inf_Model', 'Scramble_mean_real':'Scramble_Bac/Inf_Local'})
adap_entry = adap_entry[adap_entry['Gene'].notna()]
adap_entry['Phenotype'] = adap_entry['Interpret_Entry'].replace({'Block':'Inhibit', 'Uptake':'Enhance'})
non_hit = adap_entry.set_index(['Gene','Plate','Interpret_Entry']).index.tolist()

gene = list(set(non_hit) - set(hit)) # find non-hit genes 

df3 = adap_entry.set_index(['Gene','Plate','Interpret_Entry']).loc[gene,:]
df3['Hit'] = ['non-hit']*df3.shape[0]
df3['Process'] = ['Entry']*df3.shape[0]
df3['LevelShort'] = ['> 1500']*df3.shape[0]
df3['Level'] = ['Adap_Level_>1500']*df3.shape[0]
df3 = df3.drop(columns=['Plate.1', 'Batch.1','FC_Bac/Inf','Unnamed: 0', 'Batch_New'])


temp = df3.reset_index()
value = []
for g, _ in zip(temp['Gene'],temp['Plate']):
    value = value + [meanScramble.loc[_,:].values.tolist()]
Scol = ['Scramble_'+_+'_mean' for _ in meanScramble.columns]
df3 = pd.concat([df3.reset_index(),pd.DataFrame(value, columns=Scol)], axis=1).set_index(['Gene','Plate','Interpret_Entry'])

### merge three dataframe into Full_ssmd_Entry_df file
output = path+'Final_output/Full_ssmd_Entry_df.csv'
((df1.append(df2)).append(df3)).round(3).to_csv(output)

###############################################################################################################################

## 2. Bacteria Translocation process #########################################################################################
## 1. Load hit genes
trans = pd.read_csv(path+'Final_output/Final_Trans.csv', index_col=0)
hit = trans.set_index(['Gene','Plate','Interpret_Trans']).index.tolist()


################################### Conservative analysis for non-hit genes (levels > 1500) #####################################

ssmd_trans = pd.read_csv(glob.glob(path+'Conservative_method/ssmd_trans*_labelled_*.csv')[0], index_col=0)
ssmd_trans = ssmd_trans[ssmd_trans['Gene'].notna()]
ssmd_trans = ssmd_trans.rename(columns={'Gene_Symbol':'Gene', 'UnInf_cells':'Uninf_cells'})
ssmd_trans['Phenotype'] = ssmd_trans['Interpret_Trans'].replace({'Block':'Inhibit', 'Uptake':'Enhance'})
non_hit = ssmd_trans.set_index(['Gene','Plate','Interpret_Trans']).index.tolist()

gene = list(set(non_hit) - set(hit)) # find non-hit genes 

df1 = trans.set_index(['Gene','Plate','Interpret_Trans'])
df1['Hit'] = ['hit']*df1.shape[0]
df1 = df1.drop(columns=['ssmd_Trans.1','Batch.1','Translocation(Outlier)_Level'])

df2 = ssmd_trans.set_index(['Gene','Plate','Interpret_Trans']).loc[gene,:]
df2['Hit'] = ['non-hit']*df2.shape[0]
df2['Process'] = ['Trans']*df2.shape[0]
df2['LevelShort'] = ['> 1500']*df2.shape[0]
df2['Level'] = ['Con_Level_>1500']*df2.shape[0]
df2 = df2.drop(columns=['Plate.1', 'Batch.1'])

# scramble = pd.read_csv(path+'Raw_Data/Cleansing_scramble_withoutWeek.csv', index_col=0)
# meanScramble = scramble.groupby('Plate').mean()
# meanScramble['Infected/Total_cells'] = meanScramble['Inf_cells']/meanScramble['Total_cells']
# meanScramble['Nuc/IntraBac'] = meanScramble['Nuc/Inf']/meanScramble['Bac/Inf']
# meanScramble['Inf/Uninf_cells'] = meanScramble['Inf_cells']/meanScramble['UnInf_cells']

temp = df2.reset_index()
value = []
for g, _ in zip(temp['Gene'],temp['Plate']):
    value = value + [meanScramble.loc[_,:].values.tolist()]
Scol = ['Scramble_'+_+'_mean' for _ in meanScramble.columns]
df2 = pd.concat([df2.reset_index(),pd.DataFrame(value, columns=Scol)], axis=1).set_index(['Gene','Plate','Interpret_Trans'])


##################################### Adaptive analysis for non-hit genes (levels > 1500) ################################################

adap_entry = pd.read_csv(glob.glob(path+'Adaptive_RegressionModel/*Trans_ssmd*label*')[0]).drop(columns=['Week_Order', 'Week_Order_1', 'Week_Order_2','Week_Order_3'])
adap_entry = adap_entry.rename(columns={'Gene_Symbol':'Gene', 'UnInf_cells':'Uninf_cells','Scramble_Trans (Model)':'Scramble_Nuc/Inf_Model', 'Scramble_mean_real':'Scramble_Nuc/Inf_Local'})
adap_entry = adap_entry[adap_entry['Gene'].notna()]
adap_entry['Phenotype'] = adap_entry['Interpret_Trans'].replace({'Block':'Inhibit', 'Uptake':'Enhance'})
non_hit = adap_entry.set_index(['Gene','Plate','Interpret_Trans']).index.tolist()

gene = list(set(non_hit) - set(hit)) # find non-hit genes 

df3 = adap_entry.set_index(['Gene','Plate','Interpret_Trans']).loc[gene,:]
df3['Hit'] = ['non-hit']*df3.shape[0]
df3['Process'] = ['Trans']*df3.shape[0]
df3['LevelShort'] = ['> 1500']*df3.shape[0]
df3['Level'] = ['Adap_Level_>1500']*df3.shape[0]
df3 = df3.drop(columns=['Plate.1', 'Batch.1','FC_Nuc/Inf','Unnamed: 0', 'Batch_New'])


temp = df3.reset_index()
value = []
for g, _ in zip(temp['Gene'],temp['Plate']):
    value = value + [meanScramble.loc[_,:].values.tolist()]
Scol = ['Scramble_'+_+'_mean' for _ in meanScramble.columns]
df3 = pd.concat([df3.reset_index(),pd.DataFrame(value, columns=Scol)], axis=1).set_index(['Gene','Plate','Interpret_Trans'])

### merge three dataframe into Full_ssmd_Trans_df file
output = path+'Final_output/Full_ssmd_Trans_df.csv'
((df1.append(df2)).append(df3)).round(3).to_csv(output)