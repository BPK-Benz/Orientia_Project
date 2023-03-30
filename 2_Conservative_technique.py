import pandas as pd
import numpy as np
import os 
import glob
import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew, mode

# set color and background for seaborn
# colors = ['#f63770','#b937f6','#376af6','#37e3f6','#87f637','#ff7300','#f6b637','gray'] 
# sns.set_palette(colors)
# %config InlineBackend.figure_format = 'retina'
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
    # Calculate SSMD between scramble and knockdown genes by Nuc/Inf
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


    # Determine hit genes by automatic local thresholding (outlier definitons: x <= median +/- n*IQR)
    output_folder = output_folder
    for b,nb in zip([-1,1],['(Inhibit)','(Enhance)']):
        constant_threshold = []
        for t in range(100,1600,100): # target number of hit genes
            print('Process', nb , ' Plate ', t)
            number = []

            # find optimal n to get target number of hit genes
            if t < 700:
                start, end, step = [1, 2.5, 0.02]
            elif t < 1100:
                 start, end, step = [0.1, 2, 0.01]
            else:
                 start, end, step = [0.001, 2, 0.001]
                
            for n in np.arange(start, end, step).tolist():
                qs['Lower_Bound'] = qs['Median'] - n*qs['IQR']
                qs['Upper_Bound'] = qs['Median'] + n*qs['IQR']
                qs['Threshold'] = [round(max(np.abs(qs.loc[p,'Lower_Bound']),np.abs(qs.loc[p,'Upper_Bound'])),4) for p in qs.index.tolist()]
                ssmd_df['Threshold'] = [qs.loc[p,'Threshold'] for p in ssmd_df['Plate']]
                if b < 0:
                    Inhibit_gene = ssmd_df[ssmd_df[process] < b*ssmd_df['Threshold']]
                else:
                    Inhibit_gene = ssmd_df[ssmd_df[process] > b*ssmd_df['Threshold']]
                number = number + [[n, Inhibit_gene.shape[0]]]

                opt_threshold = pd.DataFrame(number, columns=['Constant','Number of gene'])
                if opt_threshold.iloc[-1,1] < t+4.5:
                    print('Thrshold: ', round(n,4))
                    # display(opt_threshold.iloc[-2:,:])
                    constant_threshold = constant_threshold + [[t, 
                                                               opt_threshold.iloc[-1:,0].tolist()[0], 
                                                               opt_threshold.iloc[-1:,1].tolist()[0]]]
                    break
        threshold = pd.DataFrame(constant_threshold, columns=['Expected_Number_gene', 'Constant', 'Exact_Number_gene'])
        threshold.to_csv(output_folder+'Threshold_'+name+'_'+nb+'.csv') # save thresholding criteria
        print('....Done...')

        for e, n in zip(threshold['Expected_Number_gene'] , threshold['Constant']):
            qs['Lower_Bound'] = qs['Median'] - n*qs['IQR']
            qs['Upper_Bound'] = qs['Median'] + n*qs['IQR']
            qs['Threshold'] = [round(max(np.abs(qs.loc[p,'Lower_Bound']),np.abs(qs.loc[p,'Upper_Bound'])),4) for p in qs.index.tolist()]
            ssmd_df['Threshold'] = [qs.loc[p,'Threshold'] for p in ssmd_df['Plate']]
            if b < 0:
                Inhibit_gene = ssmd_df[ssmd_df[process] < -1*ssmd_df['Threshold']]
            else:
                Inhibit_gene = ssmd_df[ssmd_df[process] > 1*ssmd_df['Threshold']]
            
            hit = Inhibit_gene
            hit.to_csv(output_folder+'Con_'+ name +'_Hit_genes_'+str(e)+'_'+nb+'.csv')
            print('................Done threshold...........')
            # fig,ax = plt.subplots(figsize=(10,12))
            # sns.heatmap(hit.pivot_table(index='Plate', columns='Batch', values=process, aggfunc='count',  
            #                             fill_value=0), annot=True, cmap='Reds', vmax=50, fmt='.1f' );
            # plt.title('Expected Number of Hit gene '+ nb +' : '+str(e), fontsize=15, fontweight='bold')
            # plt.show()



# set root path
path = os.getcwd()+'/../'

# # Import data
all_df = pd.read_csv(path+'Raw_Data/All_genes.csv')



output_folder = path+"Conservative_method/"
entry = conservative(dependent='Bac/Inf', process='ssmd_Entry', interpret='Interpret_Entry', output_folder=output_folder, name='Entry')
translocation = conservative(dependent='Nuc/Inf', process='ssmd_Trans', interpret='Interpret_Trans', output_folder=output_folder, name='Translocation(Outlier)')

col = ['Total_cells', 'Inf_cells', 'UnInf_cells',
       'Bac/Inf', 'Nuc/Inf', 'Cytomem/Inf', 'Infected/Total_cells',
       'Nuc/IntraBac', 'Inf/Uninf_cells']

gene = all_df[all_df['Gene_Symbol'] != 'Scramble_siRNA']
for _ in glob.glob(output_folder+'Con_*Hit*'):
    print(_)
    temp = pd.read_csv(_).set_index('Gene_Symbol')
    genelist = temp.index.tolist()
    value = gene.groupby('Gene_Symbol').mean().loc[genelist, col]
    new_df = pd.concat([value,temp], axis=1)
    new_df.to_csv(_)
    print('............Bang.......')