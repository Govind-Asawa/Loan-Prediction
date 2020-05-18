import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./loan-dataset.csv")
data = data.iloc[:,1:] #removing Loan_ID

def getUniques(data):
    dic = {}
    for col in data.columns:
        if data.loc[:,col].dtype == 'object':
            dic[col] = list(data.loc[:,col].unique())
    
    return dic

def getCounts(data): # you might also have to think abt null...
    """
        params: data (dataframe)
        
        returns:
            nested dictionary, that contains column wise value_counts 
            for each of its respective discrete categorical value
    """
    dic = {}
    for col in data.columns:
        if data.loc[:,col].dtype == 'object':
            dic[col] = dict(data.loc[:, col].value_counts()) 
    return dic

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

    # data['Income' ] = data['ApplicantIncome'] + data['CoapplicantIncome']
    # print(data.describe()[['ApplicantIncome','CoapplicantIncome','Income']])

    # data['Income_bins'] = pd.cut(data['Income'], bins = bins, labels= labels)
    # cateBiVariate(data['Income_bins'], data['Loan_Status'])
    
    '''
        NOTE: The resulting graph in such cases highly depends on the bins chosen
    '''