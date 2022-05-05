import pandas as pd
import glob
import os

# for _ in glob.glob('New*labelled*.csv'):
#     temp = pd.read_csv(_)
#     filterTemp = temp[~temp['Sample ID'].isna()]
#     filterTemp.reset_index().iloc[:,2:].to_csv('DropRedundanced_'+_)


ess_path = os.getcwd()+'/../'

finalName = 'Final_Trans.csv'

if finalName == 'Final_Entry.csv':
    enhance = pd.read_csv(ess_path+'NewData/New_Enhance_Entry_1500_labelled_3005.csv')
    enhance['Phenotype'] = ['Enhance']*enhance.shape[0]
    inhibit = pd.read_csv(ess_path+'NewData/New_Inhibit_Entry_1500_labelled_3004.csv')
    inhibit['Phenotype'] = ['Inhibit']*inhibit.shape[0]
    trans = pd.concat([inhibit,enhance],axis=0)
    trans['Entry_Level'] = inhibit['Entry_Level_1500'].tolist()+enhance['Entry_Level_1500'].tolist()
    trans = trans.drop(columns=['Entry_Level_1500','Entry_Level_1500']).iloc[:,1:]
    model = pd.read_csv(ess_path+'Adaptive_RegressionModel/Adaptive_analysis_result_Entry_ssmd_name_May_2021.csv')
    model = model[['Gene_Symbol','Scramble_mean_real','Scramble_Entry (Model)','ssmd_Entry']]
    model.columns = ['Gene_Symbol','Scramble_Bac/Inf_Local','Scramble_Bac/Inf_Model','ssmd_Entry']
    tempSelect = pd.DataFrame()
    for g in trans['Gene']:
        select = model.set_index('Gene_Symbol').loc[[g],['Scramble_Bac/Inf_Local','Scramble_Bac/Inf_Model','ssmd_Entry']]
        tempSelect = tempSelect.append(select)

else:
    enhance = pd.read_csv(ess_path+'NewData/New_Enhance_Translocation(Outlier)_600_labelled_1196.csv')
    enhance['Phenotype'] = ['Enhance']*enhance.shape[0]
    inhibit = pd.read_csv(ess_path+'NewData/New_Inhibit_Translocation(Outlier)_1500_labelled_3006.csv')
    inhibit['Phenotype'] = ['Inhibit']*inhibit.shape[0]
    trans = pd.concat([inhibit,enhance],axis=0)
    trans['Translocation(Outlier)_Level']=inhibit['Translocation(Outlier)_Level_1500'].tolist()+enhance['Translocation(Outlier)_Level_600'].tolist()
    trans = trans.drop(columns=['Translocation(Outlier)_Level_1500','Translocation(Outlier)_Level_600']).iloc[:,1:]
    model = pd.read_csv(ess_path+'Adaptive_RegressionModel/Adaptive_analysis_result_Trans_ssmd_name_May_2021.csv')
    model = model[['Gene_Symbol','Scramble_mean_real','Scramble_Trans (Model)','ssmd_Trans']]
    model.columns = ['Gene_Symbol','Scramble_Nuc/Inf_Local','Scramble_Nuc/Inf_Model','ssmd_Trans']
    tempSelect = pd.DataFrame()
    for g in trans['Gene']:
        select = model.set_index('Gene_Symbol').loc[[g],['Scramble_Nuc/Inf_Local','Scramble_Nuc/Inf_Model','ssmd_Trans']]
        tempSelect = tempSelect.append(select)

trans['Gene_Name'] = trans['Gene']
tempTrans = trans.set_index('Gene_Name')
trans = pd.concat([tempTrans,tempSelect],axis=1).reset_index()

meanScramble = pd.read_csv(ess_path+'Raw_data/Mean_Scramble_allPlate.csv').iloc[:,1:]
meanScramble['Plate'] = meanScramble['Plate'].str.replace('_P','-')

tempSelect = pd.DataFrame()
for p in trans['Plate'].tolist():
    select = meanScramble.set_index(['Plate']).loc[[p],:]
    tempSelect = tempSelect.append(select)

tempTrans = trans.set_index('Plate')
trans = pd.concat([tempTrans,tempSelect],axis=1).reset_index()
trans['LevelShort'] = [_.split('_')[-1] for _ in trans['Level']]
trans['Process'] = [finalName.split('_')[1].split('.')[0]]*trans.shape[0]
trans.drop_duplicates().drop(columns='index').to_csv(ess_path+'/NewData/'+finalName)