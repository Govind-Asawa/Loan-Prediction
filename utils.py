import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def createDummies(data, cols, drop_n_concat = True):
    if not cols:
        return data

    dummies = pd.get_dummies(data[cols], drop_first=True)
    if drop_n_concat:
        cpy = data.copy()
        cpy.drop(cols, axis = 1, inplace = True)
        cpy = pd.concat([cpy, dummies], axis=1)
        return cpy

    return dummies

def selectCategCols(data, include= None, exclude = None):
    """
        :param:
            data - Dataframe to work on
            include - the categorical columns to be included 
            exclude - the categorical columns to ignore
        :return:
            list of categorical columns 
        :raises:
            Assertion Error: if both include and exclude are provided or
                            if none of include or exclude is provided
    """
    assert include or exclude , "Atleast one parameter required"

    assert not (include and exclude), "Either of include or exclude should be provided"
    
    categ_cols = data.select_dtypes(include=["object"]).columns
    
    return [col for col in categ_cols if col in include] if include else [col for col in categ_cols if col not in exclude]

def drawROC(y_true, lr_prob):
    ns_prob = np.ones_like(y_true, dtype=np.float_)
    
    ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_prob)
    lr_fpr, lr_tpr, _ = roc_curve(y_true, lr_prob)
    
    auc_score = roc_auc_score(y_true, lr_prob)

    plt.title(f"ROC - Curve: AUC = {round(auc_score,2)}")
    plt.xlabel("False positive rate")
    plt.ylabel("true positive rate")
    plt.plot(lr_fpr, lr_tpr, label = "Logistic")
    plt.plot(ns_fpr, ns_tpr, label = "No Skill")
    plt.legend()
    plt.show()

def createSubmissionFile(ids, y_pred, filepath, threshold = 0.5 ):
    """
        creates a submission file 
        :param:
            ids - LoanID
            y_pred - the probabilities of true class
            threshold - the threshold to be used inorder to differentiate b/w the classes
            filepath - name of the output file to which the prediction is to be written (.csv)
    """

    assert len(ids) == len(y_pred), "unequal lengths of id and y_pred"
    
    if not os.path.dirname(filepath):
        filepath = ''.join(['./',filepath])

    assert os.path.exists(os.path.dirname(filepath)), "No such Directory found"

    if not y_pred.ndim == 2:
        y_pred = y_pred.reshape(-1,1)
    
    predictions = np.apply_along_axis(lambda val: 'Y' if val >= threshold else 'N', axis = 1, arr = y_pred).ravel()

    output = pd.DataFrame({'Loan_ID':ids, 'Loan_Status':predictions})

    output.to_csv(filepath, index=False)