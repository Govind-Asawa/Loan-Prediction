import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
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
    "weights": [1.8, 1]
    }

data = pd.read_csv("train.csv")
X, y = data.drop(['Loan_Status','Loan_ID'], axis =1), data.loc[:, 'Loan_Status'].map({'Y':0, 'N':1}).values

X['Total_Income'] = np.log(X['ApplicantIncome'] + X['CoapplicantIncome'])
minMaxScaler(X, ['LoanAmount', 'Loan_Amount_Term'])

X = helper(X) #this fun creates dummies i.e., one hot encoding

X = X.drop(['Gender', 'Self_Employed', 'Education', 'ApplicantIncome','CoapplicantIncome'],axis=1).values

print(f"No of features: {X.shape[1]}, No of samples: {X.shape[0]}")

log_reg = CustomLogisticRegression(**hyperparams)

pipeline = Pipeline(steps=[('i', IterativeImputer()), ('estimator',log_reg)])
pipeline.fit(X,y)

# scores = cross_val_score(pipeline, X, y, cv=5)
# print(scores, f"Mean accuracy{scores.mean()}, Standard Deviation: {scores.std()}",sep='\n')
