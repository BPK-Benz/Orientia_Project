import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


colors = ['#f63770','#b937f6','#376af6','#37e3f6','#87f637','#ff7300','#f6b637','gray'] 
sns.set_palette(colors)
# %config InlineBackend.figure_format = 'retina'
sns.set_style("white")

################################## Cleaning Data ########################################################
# set root path
path = os.getcwd()+'/../'

# # Import data
total_df = pd.read_csv(path+'Raw_Data/All_genes.csv', index_col=0)


# path = r'C:\Users\benz\phdProject\dataPreparing\ipynb\After_proposal\Essential Data/'
# total_df = pd.read_csv(path+'All_genes.csv', index_col=0)
gene = total_df[total_df['Gene_Symbol'] != 'Scramble_siRNA'].dropna()
scramble = total_df[total_df['Gene_Symbol'] == 'Scramble_siRNA'].dropna()

# 1. Scramble_siRNA has 1098 wells
## Moving outlier plate to 'Batch_10' 
Order_plate = []
for order in sorted(set(scramble['Week'].tolist())):
    min_plate = int(scramble[(scramble['Week'] == order)]['Plate'].min().split('_P')[1])
    Order_plate = Order_plate + ['P'+str(int(i.split('_P')[1])-min_plate+1) for i in scramble[(scramble['Week'] == order)]['Plate'].tolist()]
    
scramble['Order_plate'] = Order_plate

new_set = []
for ii, pp in zip(scramble['Week'].tolist(), scramble['Order_plate'].tolist()):
    if (ii >= 'Week_05-1') and (pp == 'P1'):
        w = 'Week_10'+ ii[-2:]
    else:
        w = ii
    new_set = new_set + [w]
    
scramble['New_Week'] = new_set
scramble['Batch'] = [ 'Batch_'+_.split('_')[1].split('-')[0] for _ in scramble['Week'].tolist()]
scramble['Batch_New'] = [ 'Batch_'+_.split('_')[1].split('-')[0] for _ in scramble['New_Week'].tolist()]

## Remove outlier well by outlier definition (q1-1.5*std and q3+1.5*std) 
## Observations that are smaller than the lower fence or larger than the upper fence are idetified as outlier
## Remain 1031 wells (Remove by outlier + manual) in 'Cleansing_scarmble.csv'
df = pd.DataFrame()
for r in range(2):
    for c in range(5):
        b = sorted(set(scramble['Batch_New']))
        batch = scramble[scramble['Batch_New'] == b[5*r+c]]

        q1 = lambda x: np.percentile(x, q=25)
        q3 = lambda x: np.percentile(x, q=75)
        score = batch.groupby('Plate').agg([q1, q3,'std'])['Bac/Inf']
        score.columns = ['q1','q3','std']
        for _ in score.index.tolist():
            print(_)
            moc_batch = batch[batch['Plate'] == _]
            lower = moc_batch[moc_batch['Bac/Inf'] > (score.loc[_,'q1']-1.5*score.loc[_,'std'])]
            net = lower[lower['Bac/Inf'] < (score.loc[_,'q3']+1.5*score.loc[_,'std'])]
            df = df.append(net)
        print('DF', df.shape)


# 2. Knockdown siRNA
## Moving outlier plate to 'Batch-10' like as Scramble siRNA
Order_plate = []
for order in sorted(set(gene['Week'].tolist())):
    min_plate = int(gene[(gene['Week'] == order)]['Plate'].min().split('_P')[1])
    Order_plate = Order_plate + ['P'+str(int(i.split('_P')[1])-min_plate+1) for i in gene[(gene['Week'] == order)]['Plate'].tolist()]
    
gene['Order_plate'] = Order_plate

new_set = []
for ii, pp in zip(gene['Week'].tolist(), gene['Order_plate'].tolist()):
    if (ii >= 'Week_05-1') and (pp == 'P1'):
        w = 'Week_10'+ ii[-2:]
    else:
        w = ii
    new_set = new_set + [w]
    
gene['New_Week'] = new_set

gene['Batch'] = [ 'Batch_'+_.split('_')[1].split('-')[0] for _ in gene['Week'].tolist()]
gene['Batch_New'] = [ 'Batch_'+_.split('_')[1].split('-')[0] for _ in gene['New_Week'].tolist()]

all_df = scramble.append(gene)
all_df.to_csv(path+'Adaptive_RegressionModel/All_genes_Adaptive.csv')


##########################################################################################################


################ Adaptive method ########################################################################
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit


# Define 8 formula of parent functions
def func_1(X ,a0,a1,a2,a3,a4,a5,a6): # func_1 : Multiple_Linear_Regression
    x1, x2, x3, x4, x5, x6 = X
    bb = 0
    bb += a0*np.array(x1) + a1*np.array(x2) + a2*np.array(x3) + a3
    bb += a4*np.array(x4) + a5*np.array(x5) + a6*np.array(x6)
    return bb

def func_2(X ,a0,a1,a2,a3,a4,a5,a6,a7,a8): # func_2 : Polynomial
    x1, x2, x3, x4, x5, x6 = X
    bb = 0
    bb += a0*pow(np.array(x1),3) + a1*pow(np.array(x1),2) + a2*pow(np.array(x2),3)+a3*pow(np.array(x2),2)
    bb += a4*pow(np.array(x3),3)+a5*pow(np.array(x3),2) 
    bb += a6*np.array(x4) + a7*np.array(x5) + a8*np.array(x6)
    return bb
                                                                                                        
def func_3(X ,a0,a1,a2,a3,a4,a5,a6): # func_3 : Exponential_and_Reciprocal 
    x1, x2, x3, x4, x5, x6 = X
    bb = 0
    bb += a0*np.exp(a1*np.array(x1)) + a2*1/np.array(x2)  +  a3*1/np.array(x3)
    bb += a4*np.array(x4) + a5*np.array(x5) + a6*np.array(x6)
    return bb
                                                                                                             
def func_4(X ,a0,a1,a2,a3,a4,a5,a6,a7,a8): # func_4 : Exponential2_and_Polynomial
    x1, x2, x3, x4, x5, x6 = X
    bb = 0
    bb += (a0*pow(np.array(x1),2)+a1*pow(np.array(x1),1)) + (a2*pow(np.array(x2),2)+a3*pow(np.array(x2),1))
    bb += a4*np.exp(a5*np.array(x3))
    bb += a6*np.array(x4) + a7*np.array(x5) + a8*np.array(x6) 
    return bb

def func_5(X ,a0,a1,a2,a3,a4,a5,a6,a7): # func_5 : Exponential_and_Linear 
    x1, x2, x3, x4, x5, x6 = X
    bb = 0
    bb += a0*np.exp(a1*np.array(x1)) + a2*np.array(x2) + a3*np.array(x3)+a4
    bb += a5*np.array(x4) + a6*np.array(x5) + a7*np.array(x6)
    return bb

def func_6(X ,a0,a1,a2,a3,a4,a5,a6,a7,a8): # func_6 : Exponential_and_Polynomial2_Degree21 
    x1, x2, x3, x4, x5, x6 = X
    bb = 0
    bb += a0*np.exp(a1*np.array(x2)) + (a2*pow(np.array(x1),2)+a3*pow(np.array(x1),1))
    bb += a4*pow(np.array(x3),2)+a5*pow(np.array(x3),1)
    bb += a6*np.array(x4) + a7*np.array(x5) + a8*np.array(x6)
    return bb 

def func_7(X ,a0,a1,a2,a3,a4,a5,a6,a7,a8): # func_7 : Polynomial_Degree2
    x1, x2, x3, x4, x5, x6 = X
    bb = 0
    bb += a0*pow(np.array(x1),2)+a1*pow(np.array(x1),1) + a2*pow(np.array(x2),2)+a3*pow(np.array(x2),1)
    bb += a4*pow(np.array(x3),2)+a5*pow(np.array(x3),1)
    bb += a6*np.array(x4) + a7*np.array(x5) + a8*np.array(x6)
    return bb

def func_8(X ,a0,a1,a2,a3,a4,a5,a6,a7,a8): # func_8 : Logistic
    x1, x2, x3, x4, x5, x6 = X
    bb = 0
    bb += (a0*pow(np.array(x1),2)+a1*pow(np.array(x1),1)) + (a2*pow(np.array(x2),2)+a3*pow(np.array(x2),1))
    bb += a4*np.exp(a5*np.array(x3)) 
    bb += a6*np.array(x4) + a7*np.array(x5) + a8*np.array(x6)
    return bb


num_para = [['func_1',7], ['func_2', 9], ['func_3',7], ['func_4', 9], 
            ['func_5',8], ['func_6', 9],['func_7', 9], ['func_8', 9]]

# This function is for training data set
def calculation_week(n, func, set1):
    y = set1['Bac/Inf'].tolist()
    x1 = set1['Infected/Total_cells'].tolist()
    x2 = [i/1500 for i in set1['Total_cells'].tolist()]
    x3 = [i/1500 for i in set1['UnInf_cells'].tolist()]
    x4 = set1['Week_Order_1'].tolist()
    x5 = set1['Week_Order_2'].tolist()
    x6 = set1['Week_Order_3'].tolist()
    
    
    
    popt,pcov = curve_fit(func,(x1,x2,x3, x4,x5,x6),y, maxfev=10000) # This line is difference
    y_pred = np.round(func((x1,x2,x3, x4,x5,x6), *popt),4)

    mae = mean_absolute_error(y,y_pred)
    rmse = np.sqrt(mean_squared_error(y,y_pred)) 
    r2 = r2_score(y, y_pred)
    
    return popt, [mae, rmse, r2]


# This function is to testing data set
def calculation2_week(func, popt, set1):
    y = set1['Bac/Inf'].tolist()
    x1 = set1['Infected/Total_cells'].tolist()
    x2 = [i/1500 for i in set1['Total_cells'].tolist()]
    x3 = [i/1500 for i in set1['UnInf_cells'].tolist()]
    x4 = set1['Week_Order_1'].tolist()
    x5 = set1['Week_Order_2'].tolist()
    x6 = set1['Week_Order_3'].tolist()

    
    y_pred = np.round(func((x1,x2,x3, x4,x5,x6), *popt),4)
    mae = np.round(mean_absolute_error(y,y_pred),4)
    rmse = np.round(np.sqrt(mean_squared_error(y,y_pred)),4)
    r2 = np.round(r2_score(y, y_pred),4)
    
    return [mae, rmse, r2]

def calculation2_week_value(func, popt, set1):
    y = set1['Bac/Inf'].tolist()
    x1 = set1['Infected/Total_cells'].tolist()
    x2 = [i/1500 for i in set1['Total_cells'].tolist()]
    x3 = [i/1500 for i in set1['UnInf_cells'].tolist()]
    x4 = set1['Week_Order_1'].tolist()
    x5 = set1['Week_Order_2'].tolist()
    x6 = set1['Week_Order_3'].tolist()
    y_pred = func((x1,x2,x3, x4,x5,x6), *popt)
    return y_pred

def ssmd(ss, gg):
    up = np.mean(gg)-np.mean(ss)
    down = np.power((np.power(np.std(gg),2) + np.power(np.std(ss),2)),0.5)
    r = "Block" if np.mean(gg)-np.mean(ss) < 0 else "Uptake"
    return np.round(up/down,4), r, np.mean(gg), np.std(gg), np.mean(ss), np.std(ss)

############################### Map_dictionary

parent_map ={'func_1':'Formula_1',
 'func_2':'Formula_2',
 'func_3':'Formula_3',
 'func_4':'Formula_4',
 'func_5':'Formula_5',
 'func_6':'Formula_6',
 'func_7':'Formula_7',
 'func_8':'Formula_8'}

dispatcher = { 'func_1' : func_1, 'func_2' : func_2, 
             'func_3' : func_3, 'func_4' : func_4,
             'func_5' : func_5, 'func_6' : func_6,
             'func_7' : func_7, 'func_8' : func_8}


map_dict2 = {
    'Formula_1': func_1,
    'Formula_2': func_2,
    'Formula_3': func_3,
    'Formula_4': func_4,
    'Formula_5': func_5,
    'Formula_6': func_6,
    'Formula_7': func_7,
    'Formula_8': func_8,
}

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


############################### Import data
cleaning_scramble = pd.read_csv(path+'Raw_Data/Cleansing_scramble.csv')

cleaning_scramble['Week_Order'] = [int(i[-1]) for i in cleaning_scramble['Week']]
cleaning_scramble['Week_Order_1'] = [ 1 if i == 1 else 0 for i in cleaning_scramble['Week_Order']]
cleaning_scramble['Week_Order_2'] = [ 1 if i == 2 else 0 for i in cleaning_scramble['Week_Order']]
cleaning_scramble['Week_Order_3'] = [ 1 if i == 3 else 0 for i in cleaning_scramble['Week_Order']]

total_df = pd.read_csv(path+'Adaptive_RegressionModel/All_genes_Adaptive.csv')
gene = total_df[total_df['Gene_Symbol'] != 'Scramble_siRNA']
gene['Week_Order'] = [int(i[-1]) for i in gene['Week']]
gene['Week_Order_1'] = [ 1 if i == 1 else 0 for i in gene['Week_Order']]
gene['Week_Order_2'] = [ 1 if i == 2 else 0 for i in gene['Week_Order']]
gene['Week_Order_3'] = [ 1 if i == 3 else 0 for i in gene['Week_Order']]



################################ Training and Testing model by 10 fold Cross validation
# Find the best optimal parent function for each batch by concerning (min) L2-Norm and Find the best optimal equation for each batch by (maximum) R-square
# L2-norm is used for evaluating the residual error per parameter or concerning the different of predicted and real value per parameter (Number of parameters in each model are different)
# R-square is used for evaluating the model performance of the regression models


for a in parent_map:
    func = dispatcher[a]
    function_name = parent_map[a]
    popt_all = []
    mean_var = []
    mean_r2 = []
    for batch in sorted(set(cleaning_scramble['Batch_New'])):

        data = cleaning_scramble[cleaning_scramble['Batch_New'] == batch]

        r2 = []
        rmse = []
        mae = []
        popt = []

        # prepare the cross-validation procedure
        cv = KFold(n_splits=10, random_state=9, shuffle=True) # random_state = 9,14,28,32

        ####### Training dataset ####################
        for i, (train, test) in enumerate(cv.split(data)):
 
            popt1, [mae_1, rmse_1, r2_1] = calculation_week(n=3, func=func, set1=data.iloc[train])
            popt = popt + [popt1]

        popt_all = popt_all + [[batch], popt]

        ####### Testing dataset ######################
        for j in range(10):
            for i, (train, test) in enumerate(cv.split(data)):
                [mae_2, rmse_2, r2_2] = calculation2_week(func=func, popt=popt[j], set1=data.iloc[test])
                r2 =  r2 + [['Round_'+str(j+1)+'_FC_'+str(i+1), float(r2_2)]]
                rmse =  rmse + [['Round_'+str(j+1)+'_FC_'+str(i+1),float(rmse_2)]]
                mae = mae + [['Round_'+str(j+1)+'_FC_'+str(i+1),float(mae_2)]]

        # Show 10 graph results in each batch for each parent function
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
        for ind, dff,la in zip([0,1],[r2,rmse],['R-square','Root Mean Square Error']):
            for i in range(0,20):
            #     print(i)
                if i == 0:
                    DF =  pd.DataFrame(dff[10*i:10*(i+1)])
                else:
                    DF = pd.concat([DF, pd.DataFrame(dff[10*i:10*(i+1)])], axis=1)       
            DF.columns = ['Round_'+str(int(i/2)+1) if i%2 == 0 else la+'_'+str(int(i/2)+1) for i in range(0,20)]

            if ind == 0:
                    mean_r2 = mean_r2 + [[batch]+DF.mean(axis=0).tolist()]
                    mean_var = mean_var + [[batch, la, 'Mean ', round(DF.iloc[:,1::2].mean(axis=0).max(),4), 'Fold ', np.where(DF.iloc[:,1::2].mean(axis=0) == DF.iloc[:,1::2].mean(axis=0).max())[0][0]+1, 
                                   'Variance ',round(DF.iloc[:,1::2].var(axis=0).min(),4), 'Fold ', np.where(DF.iloc[:,1::2].var(axis=0) == DF.iloc[:,1::2].var(axis=0).min())[0][0]+1]]
            else:
                    mean_var = mean_var + [[batch, la, 'Mean ', round(DF.iloc[:,1::2].mean(axis=0).min(),4), 'Fold ', np.where(DF.iloc[:,1::2].mean(axis=0) == DF.iloc[:,1::2].mean(axis=0).min())[0][0]+1, 
                                   'Variance ',round(DF.iloc[:,1::2].var(axis=0).min(),4), 'Fold ', np.where(DF.iloc[:,1::2].var(axis=0) == DF.iloc[:,1::2].var(axis=0).min())[0][0]+1]]



            DF.iloc[:,1::2].plot(kind='box', ax=ax[ind])
            ax[ind].set_xticklabels(['Fold '+str(i) for i in range(1,11)]);
            ax[ind].set_xlabel('10-Fold Cross Validation', fontweight='bold', fontsize=12);
            ax[ind].set_ylim(-0.5,1.5);
            ax[ind].set_ylabel(la, fontweight='bold', fontsize=12)
            ax[ind].set_title('Batch :'+batch, fontweight='bold', fontsize=14)
        # plt.show()
        
    experiment = pd.concat([pd.DataFrame(popt_all[::2]),pd.DataFrame(popt_all[1::2])], axis=1)
    experiment.columns = ['Batch']+['FC_'+str(i) for i in range(1,11)]

    from numpy import linalg as LA

    # L2-norm
    l2 = []
    for t in range(experiment.shape[0]):
            l2 = l2 + [LA.norm(LA.norm(experiment.iloc[t,1:], 2),2)]

    experiment['L2-Norm'] = l2

    # R-square
    R2_mean_var_df = pd.DataFrame(mean_var).iloc[::2,[0,3,5,7,9]]
    R2_mean_var_df.columns = ['Batch','R2_Mean','R2_Max_Mean_Fold','R2_Variance','R2_Min_Variance_Fold']

    # Root Mean Square Error
    RMSE_mean_var_df = pd.DataFrame(mean_var).iloc[1::2,[0,3,5,7,9]]
    RMSE_mean_var_df.columns = ['Batch','RMSE_Mean','RMSE_Min_Mean_Fold','RMSE_Variance','RMSE_Min_Variance_Fold']

    # Regression model 

    net_results = pd.concat([experiment.set_index('Batch'), R2_mean_var_df.set_index('Batch'), RMSE_mean_var_df.set_index('Batch')], axis=1)
    mean_r2_df = pd.DataFrame(mean_r2, columns=['Batch']+['Fold_'+str(i) for i in range(1,11)])
    net_results = pd.concat([net_results,mean_r2_df.set_index('Batch')], axis=1)
    net_results.to_csv(path+'Adaptive_RegressionModel/10_Replicate_net_results_model_Entry_'+function_name+'.csv')
    print('Completed process: ',func, '  ', function_name )


############################################ Evaluation model 
num_para = [['func_1',7], ['func_2', 9], ['func_3',7], ['func_4', 9], 
            ['func_5',8], ['func_6', 9],['func_7', 9], ['func_8', 9]]

b = 'L2-Norm'
l2 = []
name = []
summary = pd.DataFrame(['Batch_0'+str(i) if i < 10 else 'Batch_'+str(i) for i in range(1,11)],columns=['Batch'])
for file in glob.glob(path+"Adaptive_RegressionModel/10_Replicate_net_results_model_Entry*"):
        name = name + [file.split('_model_')[1].split('.csv')[0]]
        x = pd.read_csv(file).loc[:,b]
        l2 = l2 + [x]

dfff = pd.DataFrame(l2)
dfff.columns = ['Batch_0'+str(i) if i < 10 else 'Batch_'+str(i) for i in range(1,dfff.shape[1]+1)]
dfff = dfff.T.reset_index()
dfff.columns = [j if i == 0 else j.split('Entry_')[1] for i,j in enumerate([b] + name)]

coo = dfff.copy()
if b == 'L2-Norm':
    for i in range(1, len(dfff.columns)):
        coo.iloc[:,i] = dfff.iloc[:,i]/num_para[i-1][1]
        
print(coo.set_index('L2-Norm').style.highlight_min(color='#b3b9ff',axis=1).format("{:.4}"))


summary['Math_model'] = coo.iloc[:,1:].idxmin(axis=1)
summary['L2-Norm (min)'] = coo.iloc[:,1:].min(axis=1).round(4)
select_fold = []
r2_max = []
equation = []
graph_math = pd.DataFrame()


for order, m in enumerate(summary['Math_model']):
    k = pd.read_csv(glob.glob(path+"Adaptive_RegressionModel/10_Replicate_net_results_model_Entry*"+m+'*')[0]).iloc[:,-10:]
    graph_math = pd.concat([graph_math,k.iloc[order,:]], axis=1)
    fold = k.iloc[order,:].idxmax()
    select_fold = select_fold + [fold]
    r2_max = r2_max + [k.iloc[order,:].max().round(4)] 
    kk = pd.read_csv(glob.glob(path+"Adaptive_RegressionModel/10_Replicate_net_results_model_Entry*" + m + '*')[0]).iloc[:,1:11]
    equation = equation + [kk.iloc[order,int(fold[-1])-1]]

graph_math.columns = ['Batch_0'+str(i) if i < 10 else 'Batch_'+str(i) for i in range(1,11)]
graph_math = graph_math.T.reset_index()
graph_math['Math_model'] = summary['Math_model']
graph_math = graph_math.set_index(['index','Math_model'])
print(graph_math.style.highlight_max(color='#b3b9ff',axis=1).format("{:.4}"))

summary['Select_fold'] = select_fold
summary['R^2 (max)'] = r2_max
summary['Equation'] = equation

################################### Deployed predicted model
test_popt = summary[['Math_model','Equation']]

### Knockdown gene

model_df =  pd.DataFrame()
for order, batch in enumerate(['Batch_0'+str(i) if i < 10 else 'Batch_'+str(i) for i in range(1,11)]):
    prediction = []
    df_gene = gene[gene['Batch_New'] == batch]
    print(batch + ' Function :', map_dict2[test_popt.iloc[order, 0]])
    prediction = calculation2_week_value(func=map_dict2[test_popt.iloc[order, 0]], popt=[float(i) for i in test_popt.iloc[order,1][1:-1].split( )], set1=df_gene)
    
    df_gene['Scramble_Entry (Model)'] = prediction.tolist()
    model_df = model_df.append(df_gene)

### Scramble siRNA to confirm with predicted value and real value
model_ss_df =  pd.DataFrame()
for order, batch in enumerate(['Batch_0'+str(i) if i < 10 else 'Batch_'+str(i) for i in range(1,11)]):
    prediction = []
    df_ss = cleaning_scramble[cleaning_scramble['Batch_New'] == batch]
    print(batch + ' Function :', map_dict2[test_popt.iloc[order, 0]])
    prediction = calculation2_week_value(func=map_dict2[test_popt.iloc[order, 0]], popt=[float(i) for i in test_popt.iloc[order,1][1:-1].split( )], set1=df_ss)
    
    df_ss['Scramble_Entry (Model)'] = prediction.tolist()
    model_ss_df = model_ss_df.append(df_ss)

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

fig, ax = plt.subplots(figsize=(10,7))
sns.set_style('white')

pp = model_ss_df[(model_ss_df['Batch_New'] > 'Batch_06') & (model_ss_df['Batch_New'] < 'Batch_11')]

sns.scatterplot(data=model_ss_df, x='Bac/Inf',y='Scramble_Entry (Model)', hue='Batch_New', palette=colors)
plt.ylim(0,10)
plt.xlim(0,10)
x = range(0
          ,10)
plt.plot(x,x, 'k--')
plt.xlabel('Real data', weight='bold')
plt.ylabel('Predicted data', weight='bold')
plt.title('Scramble Comparison between Real data and Predicted data', fontsize=15, weight='bold')
# plt.show()


######################################## Hit selection with SSMD + outlier detection

ssmdvalue = []

for ii, g in enumerate(set(model_df['Gene_Symbol'].tolist())):
    ss = model_df[(model_df['Gene_Symbol'] == g)]['Scramble_Entry (Model)'].tolist()
    gg = model_df[(model_df['Gene_Symbol'] == g)]['Bac/Inf'].tolist()
    z,n,mg,sdg, ms, sds = ssmd(ss,gg)
    ssmdvalue = ssmdvalue + [[g,z,n,mg,sdg, ms, sds]]

ssmd_df_b = pd.DataFrame(ssmdvalue, columns=['Gene_Symbol','ssmd_Entry','Interpret_Entry','Mean_gene','STD_gene','Mean_scramble','STD_scramble']).set_index('Gene_Symbol')

adaptive = pd.concat([model_df.dropna().groupby('Gene_Symbol').mean(), ssmd_df_b], axis=1)

name = model_df.dropna().loc[:,['Gene_Symbol','Plate','Batch','Batch_New']].groupby('Gene_Symbol').agg(lambda x: stats.mode(x)[0][0])
sbf = pd.DataFrame(cleaning_scramble.groupby('Plate').mean()['Bac/Inf'])
name['Scramble_mean_real'] = [sbf.loc[p,'Bac/Inf'].round(4) for p in name.iloc[:,0]]

 
Z_volcano = pd.concat([adaptive, name], axis=1)
Z_volcano.to_csv(path+'Adaptive_RegressionModel/Adaptive_analysis_result_Entry_ssmd_name_May_2021.csv')
Z_volcano = Z_volcano.reset_index().sort_values(by='Plate')

qs = Z_volcano.groupby('Plate')['ssmd_Entry'].quantile([0.25,0.75])
qs = qs.unstack().reset_index()
qs.columns = ["Plate", "q1", "q3"]
qs['IQR'] = qs['q3']-qs['q1']
qs['Median'] = pd.DataFrame(Z_volcano.groupby('Plate')['ssmd_Entry'].median())['ssmd_Entry'].tolist()
qs['Batch'] = [map_dict[f] for f in qs['Plate']]
qs['Minimum'] = Z_volcano.groupby('Plate')['ssmd_Entry'].min().tolist()
qs['Maximum'] = Z_volcano.groupby('Plate')['ssmd_Entry'].max().tolist()
qs = qs.set_index('Plate')

######## plot graph to show an example of outlier detection 
fig, ax = plt.subplots(figsize=(8,14))
g = sns.boxplot(y='Plate', x='ssmd_Entry', data=Z_volcano.sort_values(by='Plate') ,dodge=False, hue='Batch', palette=colors)
number = 1.2

for o, p in enumerate(qs.index.tolist()):
        lv = number*qs.loc[p,'IQR']
        med = qs.loc[p,'Median']
        ax.plot(med+lv , o, marker='o', color='red' , markersize=5) 
        ax.plot(med-lv , o, marker='o', color='red' , markersize=5) 

plt.vlines(x=0, ymin=-2, ymax=62, colors='black', ls='--', lw=3)

plt.xlim(-10,10)
plt.legend(bbox_to_anchor=(1.1,1.0))
plt.ylabel('SSMD', fontsize=12, fontweight='bold')
plt.title('Bacteria Entry (Adaptive): Median +/- '+str(number)+'*IQR ', fontsize=12, fontweight='bold')
# plt.show()


######## Fixed number of hit genes and tuning number of n (times of IQR)
for b,nb in zip([-1,1],['(Inhibit)','(Enhance)']):
    constant_threshold = []
    for t in range(1000,700,-400):
        number = []
        for n in np.arange(0.005, 2.01, 0.001).tolist():
            qs['Lower_Bound'] = qs['Median'] - n*qs['IQR']
            qs['Upper_Bound'] = qs['Median'] + n*qs['IQR']
            qs['Threshold'] = [round(max(np.abs(qs.loc[p,'Lower_Bound']),np.abs(qs.loc[p,'Upper_Bound'])),4) for p in qs.index.tolist()]
            Z_volcano['Threshold'] = [qs.loc[p,'Threshold'] for p in Z_volcano['Plate']]
            if b < 0:
                Inhibit_gene = Z_volcano[Z_volcano['ssmd_Entry'] < b*Z_volcano['Threshold']]
            else:
                Inhibit_gene = Z_volcano[Z_volcano['ssmd_Entry'] > b*Z_volcano['Threshold']]
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
    threshold

    for e, n in zip(threshold['Expected_Number_gene'] , threshold['Constant']):
        qs['Lower_Bound'] = qs['Median'] - n*qs['IQR']
        qs['Upper_Bound'] = qs['Median'] + n*qs['IQR']
        qs['Threshold'] = [round(max(np.abs(qs.loc[p,'Lower_Bound']),np.abs(qs.loc[p,'Upper_Bound'])),4) for p in qs.index.tolist()]
        Z_volcano['Threshold'] = [qs.loc[p,'Threshold'] for p in Z_volcano['Plate']]
        if b < 0:
            Inhibit_gene = Z_volcano[Z_volcano['ssmd_Entry'] < -1*Z_volcano['Threshold']]
        else:
            Inhibit_gene = Z_volcano[Z_volcano['ssmd_Entry'] > 1*Z_volcano['Threshold']]
        
        hit = Inhibit_gene
        hit.to_csv(path+'Adaptive_RegressionModel/Adap_Entry_Hit_genes_'+str(e)+'_'+nb+'.csv')
        fig,ax = plt.subplots(figsize=(10,12))
        sns.heatmap(hit.pivot_table(index='Plate', columns='Batch', values='ssmd_Entry', aggfunc='count',  
                                    fill_value=0), annot=True, cmap='Reds', vmax=50, fmt='.1f' );
        plt.title('Expected Number of Hit gene '+ nb +' : '+str(e), fontsize=15, fontweight='bold')
        # plt.show()