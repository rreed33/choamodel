import argparse
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC  
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge 
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import dataprocessing
import math
import imblearn
# import ml_metrics as metrics

# THIS SCRIPT WILL WRITE AND SAVE ALL OF THE RELATIVE ANALYTICS FOR EACH OF THE MODELS


# INPUTS: 	FITTED MODEL, dataframe of test data from X, dataframe of test data from y
# OUTPUTS: 	METIRCS AND VISUALIZATION FOR EACH MODEL, SAVE TO ../choamodel/results/

# blah = gen_results(model, x, y)
# print(blah["AUC"]) -> returns auc_object

def main(model, X_test, y_test, model_name, file_name, group):
    y_pred = model.predict(X_test)
    print(y_pred)
    modelMetrics =	{}
    modelMetrics['Accuracy Score'] =  accuracy_score(y_test, y_pred)
    modelMetrics['Confusion Matrix'] = confusion_matrix(y_test, y_pred)  
    modelMetrics['Classification Report'] = classification_report(y_test, y_pred)
    modelMetrics['ROC AUC'] = roc_auc_score(y_test, model.predict(X_test))
    modelMetrics['Predictions'] = y_pred
    if model_name.lower() == 'log':
        coefs = sorted([(i,round(j,3)) for i , j in zip(X_test.columns, model.coef_[0])], key = lambda x: abs(x[1]), reverse = True)
        modelMetrics['coef'] = coefs

    print(modelMetrics['Accuracy Score'])
    print(modelMetrics['Confusion Matrix'])
    print(modelMetrics['Classification Report'])
    print(modelMetrics['ROC AUC'])
    # print(modelMetrics['Classification Report'])
    logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='{} (area = %{})'.format(model_name, round(logit_roc_auc,2)))
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('%s/%s_Log_ROC' %(file_name, model_name+'_'+group))

    return modelMetrics

if __name__ == '__main__':
    main()