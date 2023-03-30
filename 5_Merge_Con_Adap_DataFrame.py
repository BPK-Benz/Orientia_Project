import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings("ignore")




def put_level(process, ei, type_analysis, path):
    total_df = pd.DataFrame()
    for _ in range(100,1600,100):
        csv_files = path+type_analysis+process+'_Hit_genes_'+ str(_)+'_('+ ei +').csv'
        if os.path.exists(csv_files):
            if _ == 100:
                temp = pd.read_csv(csv_files)
                temp['Level'] = [type_analysis+'Level_'+str(_)]*temp.shape[0]
                select = temp
            else:
                temp_0 = pd.read_csv(path+type_analysis+process+'_Hit_genes_'+ str(_-100)+'_('+ ei +').csv')
                temp = pd.read_csv(csv_files)

                r = []
                one = set(temp_0['Gene_Symbol'].tolist())
                two = set(temp['Gene_Symbol'].tolist())
                for g in temp['Gene_Symbol'].tolist():
                    if g in two-one:
                        r = r + [type_analysis+'Level_'+str(_)]
                    else:
                        r = r + [type_analysis+'Level_'+str(_-100)]

                temp['Level'] =r
                select = temp[temp['Level'] == type_analysis+'Level_'+str(_)].reset_index(drop=True)

            total_df = total_df.append(select).reset_index(drop=True)

    return total_df
        
  
# 1 Merge two dataframe and put level for each gene 
# ps. Intersection gene has two rows

path = os.getcwd()+'/../'

# path = r"C:\Users\benz\phdProject\dataPreparing\ipynb\After_proposal\Output_May_2011\Final_result/"
for p in ['Translocation(Outlier)','Entry']:
    print(p)
    for x in ['Enhance','Inhibit']:
        print('Process', x)
        adapt = put_level(process=p, ei=x, type_analysis='Adap_',path = path+'Adaptive_RegressionModel/').reset_index(drop=True)
        con = put_level(process=p, ei=x, type_analysis='Con_', path=path+'Conservative_method/').reset_index(drop=True)
        a = set(adapt.index.tolist())
        b = set(con.index.tolist())
        inter = set(a) & set(b)
        merge = adapt.append(con).reset_index(drop=True)
    #     merge = merge.drop(columns=['Week_Order', 'Week_Order_1',
    #    'Week_Order_2', 'Week_Order_3','Batch_New','Scramble_mean_real'])
    #     merge = merge.drop(columns=[merge.columns[1],merge.columns[11],merge.columns[12]])
        merge.to_csv(path+'/Combination/'+x+'_'+p+'.csv')

# 2 Put Level (only adapt, only con, intersection)


for x in ['Entry','Translocation(Outlier)']:
    for y in ['Enhance_','Inhibit_']:
        csv_files = path+'Combination/'+y+x+'.csv'
#         if os.path.exists(csv_files):
        entry = pd.read_csv(csv_files).iloc[:,1:]
        entry = entry.set_index('Gene_Symbol')
        entry['Process'] = [_.split('_Level_')[0] for _ in entry['Level']]
        entry['digit'] = [int(_.split('_Level_')[1]) for _ in entry['Level']]

        for i in range(100,1600,100):
            if (x != 'Translocation(Outlier)') | (y != 'Enhance_') | (i < 700):
                print(x,y,i)
#                 entry[y+'Level_'+str(i)] = ['None']*entry.shape[0]
                adap = entry[(entry['Process'] == 'Adap') & (entry['digit'] <= i)]
                con = entry[(entry['Process'] == 'Con') & (entry['digit'] <= i)]
                a = set(adap.index.tolist())
                b = set(con.index.tolist())
                inter = set(a) & set(b)

                for g in entry.index.tolist():

                    if g in a-b:
                        entry.loc[g,x+'_Level_'+str(i)] = 'Adap_Level_'+str(i)

                    elif g in b-a:
                        entry.loc[g,x+'_Level_'+str(i)] = 'Con_Level_'+str(i)

                    elif g in inter:

                        entry.loc[g,x+'_Level_'+str(i)] = 'Inter_Level_'+str(i)

                    else:
                        entry.loc[g,x+'_Level_'+str(i)] = 'None'


        entry.to_csv(path+'Combination/'+'New_'+y+x+'.csv')


# # Double Check process?
# for x in ['Con','Adap']:
#     print(x)
#     entry = pd.read_csv(path+'Enhance_Entry.csv')
#     for i in range(100,1600,100):
#         con_100 = pd.read_csv(x+'_Entry_Hit_genes_'+str(i) + '_(Enhance).csv')
#         print(i, [i in entry['Gene_Symbol'].tolist() for i in con_100['Gene_Symbol'].tolist()].count(False))

#     entry = pd.read_csv(path+'Inhibit_Entry.csv')
#     for i in range(100,1600,100):
#         con_100 = pd.read_csv(x+'_Entry_Hit_genes_'+str(i) + '_(Inhibit).csv')
#         print(i, [i in entry['Gene_Symbol'].tolist() for i in con_100['Gene_Symbol'].tolist()].count(False))
        
#     print('\n')


# put_name
full_df = pd.read_csv(path+'Raw_Data/Full_gene_name.csv', index_col=0)
# path = 'C:/Users/benz/phdProject/dataPreparing/ipynb/After_prpathosal/'
for input_name in glob.glob(path+'Combination/'+'New*.csv'):
    output_name = input_name.split('.csv')[0].split('Combination/')[1]+'_labelled_'
    print(output_name)


###############################################################################################
    hit = pd.read_csv(input_name)
    hit=hit.rename(columns = {'Gene_Symbol':'Gene'})
    print(hit.shape[0])
    d = hit[hit['Gene'].str.startswith(('Scramble_siRNA'))].index.tolist()
    hit = hit.drop(index=d)
    print(hit.shape[0])
    hg_list = pd.DataFrame()
    count = 0

    i = 1
    for hg in hit['Gene'].tolist():
        h = hit[hit['Gene'] == hg].reset_index(drop=True)
        if hg in full_df.set_index('Gene Symbol').index.tolist():
            h_gene = full_df[full_df['Gene Symbol'] == hg].reset_index(drop=True)
            h0 = pd.concat([h,h_gene],axis=1)
            if h0.shape[0] > 1:
                count = count + 1
        else:
            if hg in full_df.set_index('Sample ID').index.tolist():
                h_gene = full_df[full_df['Sample ID'] == hg].reset_index(drop=True)
                h0 = pd.concat([h,h_gene],axis=1)
                if h0.shape[0] >1:
                    count = count + 1
            else:
                h_gene = full_df[full_df['New_name'] == hg].reset_index(drop=True)
                h0 = pd.concat([h,h_gene],axis=1)
        if i == 1:
            hg_list = h0
            i = 2
        else:
            hg_list = hg_list.append(h0)

    hg_list = hg_list.reset_index(drop=True)
    hg_list = hg_list.drop(columns=['96-well Location','Gene ID'])
    hg_list.to_csv(path+'Combination/'+output_name+str(hit.shape[0])+'.csv')