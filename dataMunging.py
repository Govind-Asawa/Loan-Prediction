import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def basicMunging(from_filepath, to_filepath=None, drop_id = False):
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

    if not os.path.dirname(from_filepath):
        from_filepath = ''.join(['./',from_filepath])

    if to_filepath and not os.path.dirname(to_filepath):
        to_filepath = ''.join(['./',to_filepath])

    assert os.path.exists(from_filepath), "No such Directory as mentioned in from_filepath"
    if to_filepath:
        assert os.path.exists(os.path.dirname(to_filepath)), "No such Directory as mentioned in to_filepath"
    
    data = pd.read_csv(from_filepath)
    if drop_id:
        data.drop(columns = ['Loan_ID'], inplace = True)

    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
    data['Married'].fillna(data['Married'].mode()[0], inplace=True)
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
    data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
    data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace = True)

    data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']

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