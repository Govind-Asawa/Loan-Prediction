import numpy as np
import pandas as pd

from collections import Counter

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from model import helper
from custom_models import CustomLogisticRegression 
from utils import calConfusionMatrix
from utils import precisionRecall
from dataMunging import minMaxScaler

hyperparams = {
    "epochs":500,
    "batch_size":72,
    "lr_rate":0.05,
    "optimizer":'rms',
    "weights": [1, 1]
    }

data = pd.read_csv("train.csv")
X, y = data.drop(['Loan_Status','Loan_ID'], axis =1), data.loc[:, 'Loan_Status'].map({'Y':0, 'N':1}).values

X['Total_Income'] = np.log(X['ApplicantIncome'] + X['CoapplicantIncome'])
minMaxScaler(X, ['LoanAmount', 'Loan_Amount_Term'])

X = helper(X) #this fun creates dummies i.e., one hot encoding

X = X.drop(['Gender', 'Self_Employed', 'Education', 'ApplicantIncome','CoapplicantIncome'],axis=1).values
# imputer = IterativeImputer()
# X = imputer.fit_transform(X)

over = SMOTE(sampling_strategy=0.85) #means the resulting minority class should have x% of no of samples of majority class
# X_res, y_res= over.fit_resample(X, y)
# print(Counter(y), Counter(y_res),sep='\n')
# X_res, y_res= under.fit_resample(X_res, y_res)
# print(Counter(y), Counter(y_res),sep='\n')

pipeline = Pipeline([('imputer',IterativeImputer()), ('over', over), ('estimator', CustomLogisticRegression(**hyperparams))])
pipeline_with_under_sampler = Pipeline([
    ('imputer',IterativeImputer()), 
    ('over', over),
    ('under',RandomUnderSampler(sampling_strategy=0.9)),#the ratio is given as Nm/NM (Nm: no of samples of minority class, NM: no of samples of majority class)
    ('estimator', CustomLogisticRegression(**hyperparams))])

scores = cross_val_score(pipeline, X, y, cv=5)
print(f"Mean accuracy: {scores.mean()} std: {scores.std()}")

scores = cross_val_score(pipeline_with_under_sampler, X, y, cv=5)
print(f"With Under sampler ---- \nMean accuracy: {scores.mean()} std: {scores.std()}")
# pipeline.fit(X,y)
# pipeline_with_under_sampler.fit(X, y)
# print(f"Normal: \n{calConfusionMatrix(y, pipeline.predict(X)[0])}")
# print(f"Under Sampling: \n{calConfusionMatrix(y, pipeline_with_under_sampler.predict(X)[0])}")


"""

observation so far

the basic oversampling with and without undersampling doesn't seem to have any impact

highest cross validation score so far : oversampling = 0.85 and undersampling = 0.9

"""