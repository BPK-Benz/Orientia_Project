import pandas as pd
import glob
import os


# set root path
path = os.getcwd()+'/../'
############################################ Evaluation model 
num_para = [['func_1',7], ['func_2', 9], ['func_3',7], ['func_4', 9], 
            ['func_5',8], ['func_6', 9],['func_7', 9], ['func_8', 9]]

b = 'L2-Norm'
l2 = []
name = []
summary = pd.DataFrame(['Batch_0'+str(i) if i < 10 else 'Batch_'+str(i) for i in range(1,11)],columns=['Batch'])
for file in glob.glob(path+"TestAdaptive/10_Replicate_net_results_model_Entry*"):
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

print(summary)

for order, m in enumerate(summary['Math_model']):
    k = pd.read_csv(glob.glob(path+"TestAdaptive/10_Replicate_net_results_model_Entry*"+m+'*')[0]).iloc[:,-10:]
    graph_math = pd.concat([graph_math,k.iloc[order,:]], axis=1)
#     print('Test', k.iloc[order,:])
    fold = k.iloc[order,:].idxmax()
    select_fold = select_fold + [fold]
    r2_max = r2_max + [k.iloc[order,:].max().round(4)] 
    kk = pd.read_csv(glob.glob(path+"TestAdaptive/10_Replicate_net_results_model_Entry*" + m + '*')[0]).iloc[:,1:11]
    equation = equation + [kk.iloc[order,int(fold[-1])-1]]

    print(pd.DataFrame([1,2,3,4]))