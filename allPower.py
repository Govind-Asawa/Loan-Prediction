import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbPipeline

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

from dataMunging import minMaxScaler
from utils import createSubmissionFile

def factorizing(X, cols):
    data = X.copy()
    for col in cols:
        data[col], _ = pd.factorize(data[col])
        data[col].replace(-1, np.nan, inplace=True)
        data[col] += 1
    
    return data

def oneHot(data, cols):
    return pd.get_dummies(data, columns=cols, drop_first=True)

def makeConfusionMatrix(y_true, y_pred, threshold=0.5):
    return confusion_matrix(y_true, (y_pred >= threshold).astype(int))

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

estimators = {
    'Log_reg': LogisticRegression(),
    
    'RandomForest':RandomForestClassifier(n_estimators=50, 
                   min_samples_split=25,
                   min_samples_leaf=10,criterion='gini',random_state=0,class_weight={0:1., 1:1.5}),
    
    'Bagging': BaggingClassifier(RandomForestClassifier(n_estimators=50, 
                                                        min_samples_split=25,
                                                        min_samples_leaf=10,
                                                        criterion='gini',
                                                        random_state=0,
                                                        class_weight={0:1., 1:1.5})),
    
    'AdaBoost': AdaBoostClassifier(RandomForestClassifier(n_estimators=50, 
                                                        min_samples_split=25,
                                                        min_samples_leaf=10,
                                                        criterion='gini',
                                                        random_state=0,
                                                        class_weight={0:1., 1:1.5}), random_state=1, n_estimators=10, learning_rate=0.05),

    
    'GBM': GradientBoostingClassifier(n_estimators=20,
                                      random_state=5,
                                      learning_rate=0.05,
                                      min_samples_split=30,
                                      min_samples_leaf=5,
                                      max_features = 'auto'
                                     )
}

def prepare(X):
    X['Total_Income'] = X['ApplicantIncome'] + X['CoapplicantIncome']
    X['EMI'] = X['LoanAmount']/X['Loan_Amount_Term'] 
    X['left_per_month'] = X['Total_Income'] - X['EMI']*1000
    X.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1, inplace=True)
    X = factorizing(X, ['Dependents', 'Property_Area', 'Education'])
    X = oneHot(X, ['Married','Gender', 'Self_Employed'])
    minMaxScaler(X, ['LoanAmount', 'Loan_Amount_Term', 'Total_Income', 'left_per_month'])
    return X

X, y = train_data.drop(['Loan_Status','Loan_ID'], axis=1), train_data['Loan_Status'].map({'Y':0, 'N':1})
test_X, ids = test_data.drop(['Loan_ID'], axis=1), test_data['Loan_ID']
X = prepare(X)
test_X = prepare(test_X)

pipeline = Pipeline([('imputer',IterativeImputer()),
                        ('estimator', VotingClassifier([
                            ('randforest',estimators['RandomForest']),
                            ('adaboost',estimators['AdaBoost']),
                            ('Bagging', estimators['Bagging'])
                        ],voting='soft'))])

pipeline.fit(X, y)
y_pred = pipeline.predict_proba(test_X)[:, 1]
createSubmissionFile(ids, y_pred, "Voting_Strategy.csv", threshold=0.4)
