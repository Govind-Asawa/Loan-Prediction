import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def cut(feature, labels, bins = None):
    """
        generates a categorical view of the feature
        The function is smart enough to categorize the feature based on its quantiles
        if bins are not passed
        
        :param:

            feature - Series or 1D array on Which cut operation is to be applied
            labels - list of names to be given to the partitions
            bins - options , used if given else follows quantile parition system
        
        :return:
            categorical view of the feature as 1D-array

        for eg:
            Income = pd.Series(np.random.rand(1000)) 
            cut(Income, ['low', 'moderate','high'])
            it would divide the feature such that 
                Income(min - quantile(0.33))            = 'low'
                Income(quantile(0.33) - quantile(0.66)) = 'moderate'
                Income(quantile(0.66) - max)            = 'high
    """

    assert len(labels) >= 2, "No. of labels should be atleast 2"

    if bins:
        assert len(bins) == len(labels) + 1, "Length mismatch"
    
    feature = np.array(feature)

    if not bins:
        bins = [feature.min(), feature.max()]
        
        n_quantiles = len(labels)-1
        quantile_val = round(1./len(labels), 2)

        for i in range(1, n_quantiles+1):
            bins.insert(i, np.quantile(feature, quantile_val*i))
    
    return np.array(pd.cut(feature, bins, labels=labels))

def testFiles(from_filepath, to_filepath = None):
    """
        utility function to test existence of files passed
    """

    if not os.path.dirname(from_filepath):
        from_filepath = ''.join(['./',from_filepath])

    if to_filepath and not os.path.dirname(to_filepath):
        to_filepath = ''.join(['./',to_filepath])

    assert os.path.exists(from_filepath), "No such Directory as mentioned in from_filepath"
    if to_filepath:
        assert os.path.exists(os.path.dirname(to_filepath)), "No such Directory as mentioned in to_filepath"
    
    return from_filepath, to_filepath

def ImputeC_H(data, thresh = 0.5):
    
    def fun(x):
        ones = dict(x.value_counts(normalize = True)).get(1.0, 0)
        return 1.0 if ones >= thresh else 0.0
    
    table = pd.pivot_table(data, index = ['TotalIncome_cat'], values=['Credit_History'], columns=['Property_Area', 'Dependents'], aggfunc=fun)['Credit_History']
    data['Credit_History'].fillna(data[data['Credit_History'].isnull()].apply(
        lambda row:
        table[row['Property_Area']][row['Dependents']][row['TotalIncome_cat']],
        axis = 1
    ), inplace = True )

def minMaxScaler(data, cols):
    """
        scales numeric data
    """
    for col in cols:
        data[col] = (data[col] - data[col].min())/(data[col].max() - data[col].min())

def basicMunging(from_filepath, to_filepath=None, drop_id = False, imputeCredit_History = True):
    """
        performs basic data munging i.e.,
        fills na values in Categorical variables with their respective mode values
        and add a new column such that

        TotalIncome_log = log(ApplicantIncome + CoapplicantIncome)

        log is an accepted trick to fix outliners and normalize the data

        :param:
            from_filepath - path to loan-dataset 
            to_filepath - path a csv file where the result is to be written, if specified
            drop_id - specifies whether or not to remove Loan_ID column from the dataset read

        :return:
            returns processed data if to_filepath is not specified else returns to_filepath

        :raises:
            AssertionError : if from_filepath doesn't exist
                            and also if to_filepath's directory doesn't exist
    """

    from_filepath, to_filepath = testFiles(from_filepath, to_filepath)

    data = pd.read_csv(from_filepath)
    if drop_id:
        data.drop(columns = ['Loan_ID'], inplace = True)

    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
    data['Married'].fillna(data['Married'].mode()[0], inplace=True)
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
    data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
    data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace = True)

    data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    
    if imputeCredit_History:
        data['TotalIncome_cat'] = cut(data['TotalIncome'], labels=['low', 'moderate', 'high', 'very high'])
        ImputeC_H(data, thresh=0.6)
        data.drop('TotalIncome_cat', axis = 1, inplace = True)
        
    else:
        data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)

    
    data['TotalIncome_log'] = np.log(data['TotalIncome']) #log transform is a trick to fix skwed distributions and outliners
    data.drop(['ApplicantIncome','CoapplicantIncome','TotalIncome'], axis = 1, inplace = True)
    # uncomment to test the impact of np.log
    """
    # the title includes coefficient of variation = std(x)/mean(x)

    fig, axs = plt.subplots(1,2)
    fig.figsize = (8,8)

    axs[0].set_title(f"Actual distribution {round(data['TotalIncome'].std()/data['TotalIncome'].mean(),2)}")
    axs[0].boxplot(data['TotalIncome'])
    axs[1].set_title(f"log distribution {round(data['TotalIncome_log'].std()/data['TotalIncome_log'].mean(),2)}")
    axs[1].boxplot(data['TotalIncome_log'])
    plt.show()
    """

    data['Dependents'].replace('3+','3', inplace = True)
    if to_filepath:
        data.to_csv(to_filepath, index = False)
        return to_filepath
    else:
        return data
