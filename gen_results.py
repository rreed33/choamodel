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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import dataprocessing
import math
import imblearn
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.cross_validation import cross_val_predict
# import ml_metrics as metrics

# THIS SCRIPT WILL WRITE AND SAVE ALL OF THE RELATIVE ANALYTICS FOR EACH OF THE MODELS


# INPUTS: 	FITTED MODEL, dataframe of test data from X, dataframe of test data from y
# OUTPUTS: 	METIRCS AND VISUALIZATION FOR EACH MODEL, SAVE TO ../choamodel/results/

# blah = gen_results(model, x, y)
# print(blah["AUC"]) -> returns auc_object

def main(model, X_test, y_test, model_name, file_name, group, cv):
    y_pred = model.predict(X_test)
    # print(y_pred)
    modelMetrics =	{}
    modelMetrics['Accuracy Score'] =  accuracy_score(y_test, y_pred)
    modelMetrics['Confusion Matrix'] = confusion_matrix(y_test, y_pred)  
    modelMetrics['Classification Report'] = classification_report(y_test, y_pred)
    modelMetrics['ROC AUC'] = roc_auc_score(y_test, model.predict(X_test))
    modelMetrics['Predictions'] = y_pred
    if model_name.lower() == 'log':
        coefs = sorted([(i,round(j,3)) for i , j in zip(X_test.columns, model.coef_[0])], key = lambda x: abs(x[1]), reverse = True)
        print(coefs)
        modelMetrics['coef'] = coefs

    print(modelMetrics['Accuracy Score'])
    print(modelMetrics['Confusion Matrix'])
    print(modelMetrics['Classification Report'])
    print(modelMetrics['ROC AUC'])

    if cv > 0:
        Show_Precision = make_scorer(precision_score, pos_label = 0)
        NoShow_Precision = make_scorer(precision_score, pos_label = 1)
        Show_Recall = make_scorer(recall_score, pos_label = 0)
        NoShow_Recall = make_scorer(recall_score, pos_label = 1)
        Show_F1 = make_scorer(recall_score, pos_label = 0)
        NoShow_F1 = make_scorer(recall_score, pos_label = 1)
        # scoring = ['accuracy', 'precision', 'recall', 'f1']
        scoring = {'Show_Precision': Show_Precision,
                    'NoShow_Precision': NoShow_Precision,
                    'Show_Recall': Show_Recall,
                    'NoShow_Recall': NoShow_Recall,
                    'Show_F1': Show_F1,
                    'NoShow_F1': NoShow_F1}
        kfold = StratifiedKFold(n_splits=cv, random_state=42, shuffle = True)
        modelMetrics['CV Scores'] = cross_validate(estimator=model,
                                          X=X_test,
                                          y=y_test,
                                          cv=kfold,
                                          scoring=scoring)
        metrics = ['train_Show_Precision', 'test_Show_Precision', 'train_NoShow_Precision', 'test_NoShow_Precision',
                    'train_Show_Recall', 'test_Show_Recall', 'train_NoShow_Recall', 'test_NoShow_Recall',
                    'train_Show_F1', 'test_Show_F1', 'train_NoShow_F1', 'test_NoShow_F1']
        print('CV metrics - array, avg, std')
        for i in metrics:
            arr = modelMetrics['CV Scores'][i]
            print()
            print(i)
            print('Array: ', arr)
            print('Average: ', np.mean(arr))
            print('Std: ', np.std(arr))

    #y_pred2 = cross_val_predict(model, X_test, y_test, cv=10)
    ##print(classification_report(y_test, y_pred2))

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