import numpy as np
import pandas as pd
from custom_models import CustomLogisticRegression 
from model import helper
from sklearn.model_selection import train_test_split
from utils import calConfusionMatrix, precisionRecall, createSubmissionFile
from sklearn.linear_model import LogisticRegression
from dataMunging import minMaxScaler, basicMunging

_ = basicMunging("train.csv", "processed-train.csv", drop_id=True, imputeCredit_History=True)

data = pd.read_csv("processed-train.csv")
data = helper(data, ['Gender', 'Self_Employed', 'Education']+['Loan_Status'])
data['Loan_Status'] = data['Loan_Status'].map({'Y':0, 'N':1})

X, y = data.drop(['Gender', 'Self_Employed', 'Education']+['Loan_Status'], axis =1), data['Loan_Status']

minMaxScaler(X, ['LoanAmount', 'Loan_Amount_Term'])

epochs, batch_size, lr_rate, optimizer, weights = 500, 72, 0.05, "rms", [1.8, 1]

log_reg = CustomLogisticRegression(epochs=epochs, batch_size=batch_size, lr_rate=lr_rate, optimizer=optimizer, weights=weights)
log_reg.fit(X, y, verbose = 2)
y_pred = log_reg.predict(X)[0]
print(calConfusionMatrix(y, y_pred))
precisionRecall(y, y_pred)

test_data = pd.read_csv("test.csv")
test_data = basicMunging("test.csv", drop_id=False, imputeCredit_History=True)
ids = test_data['Loan_ID']
test_data = helper(test_data, ['Gender', 'Self_Employed', 'Education', 'Loan_ID'])
test_data.drop(['Gender', 'Self_Employed', 'Education', 'Loan_ID'], axis=1, inplace=True)
minMaxScaler(test_data, ['LoanAmount', 'Loan_Amount_Term'])

y_prob = log_reg.predict(test_data)[0]
createSubmissionFile(ids, y_prob, "custom_model.csv", threshold=0.4)