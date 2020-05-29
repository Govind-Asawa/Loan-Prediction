import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import utils
import dataMunging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def helper(data, cols_to_remove = ['Gender', 'Self_Employed', 'Education']):
    
    cpy = data.copy()
    cpy.loc[:,'Dependents'] = cpy.loc[:,'Dependents'].astype("object")

    cols_of_interest = utils.selectCategCols(cpy, exclude= cols_to_remove)
    data_to_feed = utils.createDummies(cpy, cols_of_interest)
    return data_to_feed

def trainDecisionTree(data, cols_to_remove = ['Gender', 'Self_Employed', 'Education'],val_size = None):
    
    data_to_feed = helper(data, cols_to_remove + ['Loan_Status'])
    data_to_feed.loc[:,'Loan_Status'] = data_to_feed.loc[:, 'Loan_Status'].map({'Y':0 , 'N':1}).astype('uint8')

    train_X, train_y = data_to_feed.drop(cols_to_remove + ['Loan_Status'], axis = 1), data_to_feed.loc[:,'Loan_Status']

    tree = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=500, random_state=2, min_samples_split=25)
    
    if val_size:
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = val_size, random_state = 1, shuffle = True)
    
    tree.fit(train_X, train_y)

    if val_size:
        return tree, (val_X, val_y)
    
    return tree


def trainModel(train_data, cols_to_remove = ['Gender', 'Self_Employed', 'Education'], val_size = None):
    """
        performs the basic operations required to setup a preprocessed dataset inorder to start predicting
        
        :param:
            data - The Loan dataset on which LogisticRegression model is to be fit
            cols_to_remove - the categorical cols which are not to be considered for creating dummies 
            val_size - If none total data is used to train the model or 
            [0., 1) amount of data to be set apart for validation

        :return:
            returns
            if val_size is provided then 
                trained LogisticRegressor, (X_val, y_val)
            else
                trained LogisticRegression
    """
    data_to_feed = helper(train_data, cols_to_remove + ['Loan_Status'])

    data_to_feed['Loan_Status'] = data_to_feed.loc[:, 'Loan_Status'].map({'Y':0,'N':1}).astype('uint8')

    X_train, y_train = data_to_feed.drop(cols_to_remove + ['Loan_Status'],axis=1), data_to_feed.loc[:, 'Loan_Status']
    
    if val_size:
        X_train, X_val, y_train, y_val = train_test_split(X, data_to_feed.loc[:,'Loan_Status'], random_state=0, test_size = val_size)

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    
    if val_size:
        return log_reg, (X_val, y_val)
    
    return log_reg

def prepareTestData(cols_to_remove):
    test_data = dataMunging.basicMunging("test.csv", drop_id = False, imputeCredit_History=True)
    test_data = helper(test_data, cols_to_remove + ['Loan_ID'])

    ids = test_data.loc[:,'Loan_ID']

    return ids, test_data.drop(cols_to_remove+['Loan_ID'], axis = 1)


# processing and reading --------

cols_to_remove = ['Gender', 'Self_Employed', 'Education']

# _ = dataMunging.basicMunging("train.csv", "processed-train.csv", drop_id=True, imputeCredit_History=True)

data = pd.read_csv("processed-train.csv")
tree = trainDecisionTree(data, cols_to_remove = cols_to_remove)
ids, test_data = prepareTestData(cols_to_remove)
y_prob = tree.predict_proba(test_data)[:,1]
utils.createSubmissionFile(ids, y_prob, "submission_dt.csv", threshold = 0.4)
'''
log_reg = trainModel(data, cols_to_remove = cols_to_remove)

y_pred = log_reg.predict_proba(test_data)[:,1]
utils.createSubmissionFile(ids, y_pred, "submission2_reversed.csv", threshold=0.4)
'''