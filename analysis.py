import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./loan-dataset.csv")
data = data.iloc[:,1:] #removing Loan_ID

def getUniques(data):
    return {col:list(data[col].unique()) for col in data.select_dtypes(include="object").columns}

def getCounts(data): # you might also have to think abt null...
    """
        params: data (dataframe)
        
        returns:
            nested dictionary, that contains column wise value_counts 
            for each of its respective discrete categorical value
    """
    return {col:dict(data[col].value_counts()) for col in data.select_dtypes(include="object").columns}

def continuousUniVariate(feature, bins = 50):
    
    if feature.dtype == 'object':
        return
    
    fig, axs = plt.subplots(1,2)
    axs[0].boxplot(feature)
    axs[1].hist(feature, bins=bins) #, histtype='barstacked'
    plt.show()

def categUniVariate(feature, normalize = True):

    if feature.dtype != 'object':
        return 
    
    fig = plt.figure(figsize=(8,4))
    feature.value_counts(normalize = normalize).plot(kind="bar", title = feature.name)
    plt.show()

def cateBiVariate(feature, target):
    """
        function plots a bar diag to show the relationship of feature and target 
    """
    counts = pd.crosstab(index = feature, columns=target)
    title = f'{feature.name} Vs {target.name}'
    counts.div(counts.sum(1), axis=0).plot(kind='bar', stacked = True, figsize = (4,4), title = title)
    plt.show()

def mode(x):
    return x.mode()

def comprehensiveMode(x):
    dic = dict(x.value_counts(normalize = True))
    items = sorted(dic.items(), key= lambda x: x[1] , reverse = True)
    class_val = [f'{item[0]}: {round(item[1],2)*100} %' for item in items[:2]]
    return '  '.join(class_val) + f" {x.shape[0]}"

def fun(x):
    ones = dict(x.value_counts(normalize = True)).get(1.0, 0)
    return 1.0 if ones >= 0.7 else 0.0


if __name__ == '__main__':

    # for col in [col for col in data.columns if data[col].dtype != 'object']:
    #     temp = data.loc[:,col]
    #     print(f'{col}: {temp.isna().sum()}')
    #     time.sleep(1)
        # continuousUniVariate(temp) plots graphs

    # for col in data.columns:
    #     categUniVariate(data.loc[:,col])

    # cateBiVariate(data['Gender'], data['Loan_Status'])

    # Continuous var

    # --- yeilds no great results ---
    # data.groupby('Loan_Status')['ApplicantIncome'].mean().plot('bar') 

    bins = [0, 2800, 4000, 6000, 81000] #creating bins based on quartiles observed from boxplot
    labels = ['low', 'moderate', 'high', 'very high']

    # data['AIncome_bin'] = pd.cut(data['ApplicantIncome'], bins = bins, labels=labels)
    # cateBiVariate(data['AIncome_bin'], data['Loan_Status'])
    
    # plt.show()

    data['Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    # print(data.describe()[['ApplicantIncome','CoapplicantIncome','Income']])

    data['Income_bin'] = pd.cut(data['Income'], bins = bins, labels= labels)
    # cateBiVariate(data['Income_bins'], data['Loan_Status'])
    
    # ------part of iteration 2
    # for col in data.select_dtypes('object').columns:
    #     temp = data.groupby(col)['Credit_History'].agg([comprehensiveMode])
    #     print(temp)
    
    # temp = data.groupby(['Education', 'Property_Area', 'Dependents'])['Credit_History'].agg([comprehensiveMode])
    # temp = data.groupby(['Income_bin', 'Property_Area', 'Dependents'])['Credit_History'].agg([comprehensiveMode])
    # print(temp)
    def timepass(x):
        return table[x['Property_Area']][x['Dependents']][x['Income_bin']]
    
    table = pd.pivot_table(data, index = ['Income_bin'], values=['Credit_History'], columns=['Property_Area', 'Dependents'], aggfunc=fun)['Credit_History']
    
    print(data[data['Credit_History'].isnull()].apply(timepass, axis = 1))
    '''
        NOTE: The resulting graph in such cases highly depends on the bins chosen
    '''