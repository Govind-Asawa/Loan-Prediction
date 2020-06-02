import numpy as np
import pandas as pd
from custom_models import CustomLogisticRegression 
from model import helper
from sklearn.model_selection import train_test_split
from utils import calConfusionMatrix, precisionRecall
from sklearn.linear_model import LogisticRegression
from dataMunging import minMaxScaler

data = pd.read_csv("processed-train.csv")
data = helper(data, ['Gender', 'Self_Employed', 'Education']+['Loan_Status'])
data['Loan_Status'] = data['Loan_Status'].map({'Y':0, 'N':1})
X, y = data.drop(['Gender', 'Self_Employed', 'Education']+['Loan_Status'], axis =1), data['Loan_Status']

minMaxScaler(X, ['LoanAmount', 'Loan_Amount_Term'])

train_X, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 1)


actual_log_reg = LogisticRegression()
actual_log_reg.fit(train_X, train_y)
actual_y_prob = actual_log_reg.predict_proba(test_x)[1]

epochs, batch_size, lr_rate, optimizer, weights = 80, 50, 0.09, "rms", [1.8, 1]

log_reg = CustomLogisticRegression(epochs=epochs, batch_size=batch_size, lr_rate=lr_rate, optimizer=optimizer, weights=weights)
log_reg.fit(train_X, train_y, verbose = 2)
# print("Actual coef_",actual_log_reg.coef_,sep='\n')

# print("My coef_",log_reg.coef_,sep='\n')

y_prob = log_reg.predict(test_x)
print(f"\nepochs: {epochs}, batch_size: {batch_size}, lr_rate: {lr_rate}, optimizer: {optimizer}, weights: {weights}")
print(calConfusionMatrix(test_y, y_prob[0], threshold=0.5))
# precisionRecall(test_y, y_prob[0])