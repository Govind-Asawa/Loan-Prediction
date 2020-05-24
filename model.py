import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import utils
import dataMunging
from sklearn.model_selection import train_test_split

def helper(data, cols_to_remove = ['Gender', 'Self_Employed', 'Education']):
    
    cpy = data.copy()
    cpy.loc[:,'Dependents'] = cpy.loc[:,'Dependents'].astype("object")

    cols_of_interest = utils.selectCategCols(cpy, exclude= cols_to_remove)
    data_to_feed = utils.createDummies(cpy, cols_of_interest)
    return data_to_feed


def trainModel(train_data, cols_to_remove = ['Gender', 'Self_Employed', 'Education'], val_size = 0.2):
    """
        performs the basic operations required to setup a preprocessed dataset inorder to start predicting
        
        :param:
            data - The Loan dataset on which LogisticRegression model is to be fit
            cols_to_remove - the categorical cols which are not to be considered for creating dummies 
            val_size - [0., 1) amount of data to be set apart for validation

        :return:
            returns 
            trained LogisticRegressor, (X_val, y_val)
    """
    data_to_feed = helper(train_data, cols_to_remove + ['Loan_Status'])

    data_to_feed['Loan_Status'] = data_to_feed.loc[:, 'Loan_Status'].map({'Y':1,'N':0}).astype('uint8')

    X = data_to_feed.drop(cols_to_remove + ['Loan_Status'],axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, data_to_feed.loc[:,'Loan_Status'], random_state=0, test_size = val_size)
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    return log_reg, (X_val, y_val)

# processing and reading --------

cols_to_remove = ['Gender', 'Self_Employed', 'Education']

path = dataMunging.basicMunging("train.csv", "processed-train.csv", drop_id=True)

data = pd.read_csv(path)
# --------

# train data ----
log_reg, (X_val, y_val) = trainModel(data, cols_to_remove= cols_to_remove)
# lr_prob = log_reg.predict_proba(X_val)[:,1]
# utils.drawROC(y_val, lr_prob)

# test data-----

test_data = dataMunging.basicMunging("test.csv", drop_id = False)
test_data = helper(test_data, cols_to_remove + ['Loan_ID'])

ids = test_data.loc[:,'Loan_ID']
test_data.drop(cols_to_remove+['Loan_ID'], axis = 1, inplace = True)

y_pred = log_reg.predict_proba(test_data)[:,1]
utils.createSubmissionFile(ids, y_pred, "submission.csv", threshold=0.3)