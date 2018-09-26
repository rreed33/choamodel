import argparse
import sklearn
import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC  
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import math
import imblearn
import os
import datetime as dt 

import dataprocessing
from gen_results import main as make_results


def record_file(args, df):
    parameter_names = ['Test', 'Hot', 'Group', 'NoCancel', 'Over']
    parameters      = [args.test_size, args.one_hot, args.group, args.no_cancel, args.sample_type]
    file_name   = '../choamodel/results/'
    file_ext     = '_'.join([str(i)+'_'+str(j) for i,j in zip(parameter_names, parameters)]) +'_API_True'
    file_name = file_name + file_ext + '/'

    print(file_name)
    if not os.path.exists(file_name):
        print('made')
        os.makedirs(file_name)

    with open(file_name+'results.txt', 'w') as f:
        f.write('---This file was generate as a part of Senior Design by team 14 on {} ---\n'.format(dt.datetime.today()))
        f.write('\nCOMMAND LINE ARGUMENTS:\n '+', '.join([str(i)+': '+str(j) for i, j in zip(parameter_names, parameters)]))
        f.write('\nTraining Group:\t{}\nNumber of Encounters: {}\nNumber of Patients: {}\nNumber of Features: {}'.format(args.group.upper(),len(df), len(np.unique(df['Sibley_ID'])), len(df.columns)))
        f.write('\nNumber of No Shows:\t{}\n'.format(sum(df['No_Show'])))
        f.write('\nFeature Names:\n{}\n'.format(', '.join([i for i in df.keys()])))
    return file_name

#unique patients
#arguments 
#number of encounters
#features

#converts the classification report into a dataframe
def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)

    return dataframe

def record_results(model, results, file_name, group):

    with open(file_name+'results.txt', 'a') as f:
        f.write('\n\n----------Writing Results for __{}_{}__ ----------\n'.format(model, group))
        f.write('Accuracy Score:\t{}\n'.format(results['Accuracy Score']))
        f.write('Confusion Matrix:\n{}\n'.format(results['Confusion Matrix']))
        f.write('Classification Report:\n{}\n'.format(results['Classification Report']))
        f.write('ROC AUC Score:\t{}\n'.format(results['ROC AUC']))


def main(args):
    #EVERYTHING ABOVE HERE CAN BE IGNORED
    df = dataprocessing.main(args.group, args.no_cancel, args.one_hot)

    df['Payor_Type_ID'].astype(int).astype('category')
    df['Dept_ID'].astype('category')
    df['Provider_ID'].astype('category')
    df['Appt_Logistics_Type_ID'].astype('category')
    df['Visit_Type_ID'].astype('category')

    #split the data into dependent and predictor
    X = df.drop(['No_Show','Sibley_ID'], axis=1)  
    y = df['No_Show']
    print(X.dtypes)

    #record initial metrics about the dataset
    file_name = record_file(args, df)

    # over sampling the no show class to even class distribution
    # RYAN: over_sample will be a list of the inputs you write, indexed according to imput order, first being model method type
    if args.sample_type == 'overunder':
        # args.over_sample[1:] are the parameters to plug into 
        from imblearn.combine import SMOTETomek
        smt = SMOTETomek(ratio='auto')
        X, y = smt.fit_sample(X, y)

    elif args.sample_type == 'underTomek':
        from imblearn.under_sampling import TomekLinks
        tl = TomekLinks(return_indices=True, ratio='majority')
        X, y = tl.fit_sample(X, y)

    elif args.sample_type == 'underCentroid':
        from imblearn.under_sampling import ClusterCentroids
        cc = ClusterCentroids(ratio={0: 10})
        X, y = cc.fit_sample(X, y)

    elif args.sample_type == 'overSMOTE':
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(ratio='minority')
        X, y = smote.fit_sample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state = 1001)

    #build the model
    print('='*20)
    print('INITIALIZING MODELS')

    # , 'kernel_ridge' causes a problem with memory for rn
    # , 'ridge_reg', 'lasso_reg' causes a  problem with binary and continuous target space
    model_types = ['log', 'dtree', 'rf', 'knn']
    for model in model_types:
        #start a classifier
        if model == 'SVM':
            print('-'*20)
            print('initializing {} model'.format(model))
            classifier = SVC()
            print('-'*10)
            print('fitting {} model'.format(model))
            classifier.fit(X_train, y_train)
        elif model == 'log':
            print('-'*10)
            print('initializing {} model'.format(model))
            classifier = LogisticRegression()
            print('-'*10)
            print('fitting {} model'.format(model))
            classifier.fit(X_train, y_train)
        elif model == 'dtree':
            print('-'*10)
            print('initializing {} model'.format(model))
            classifier = DecisionTreeClassifier()
            print('-'*10)
            print('fitting {} model'.format(model))
            classifier.fit(X_train, y_train)
        elif model == 'rf':
            print('-'*10)
            print('initializing {} model'.format(model))
            classifier = RandomForestClassifier(n_estimators=100)
            print('-'*10)
            print('fitting {} model'.format(model))
            classifier.fit(X_train, y_train)
        elif model == 'kernel_ridge':
            print('-'*10)
            print('initializing {} model'.format(model))
            classifier = KernelRidge()
            print('-'*10)
            print('fitting {} model'.format(model))
            classifier.fit(X_train, y_train)
        elif model == 'ridge_reg':
            print('-'*10)
            print('initializing {} model'.format(model))
            classifier = Ridge()
            print('-'*10)
            print('fitting {} model'.format(model))
            classifier.fit(X_train, y_train)
        elif model == 'lasso_reg':
            print('-'*10)
            print('initializing {} model'.format(model))
            classifier = Lasso()
            print('-'*10)
            print('fitting {} model'.format(model))
            classifier.fit(X_train, y_train)
        elif model == 'knn':
            print('-'*10)
            print('initializing {} model'.format(model))
            classifier = KNeighborsClassifier()
            print('-'*10)
            print('fitting {} model'.format(model))
            classifier.fit(X_train, y_train)

        print('GENERAL')
        results = make_results(classifier, X_test, y_test, model, file_name)
        record_results(model, results, file_name, args.group.upper())

        if args.group == 'all':
            #split the ALL segment into HISTORICAL AND NONHISTORICAL
            hist_mask = X_test['count_app'] >= 1
            nonhist_mask = X_test['count_app'] == 0

            #write resutls for HISTORICAL SEGMENT
            print('HISTORICAL')
            results_hist = make_results(classifier, X_test[hist_mask], y_test[hist_mask], model, file_name)
            record_results(model, results_hist, file_name, 'HISTORICAL')

            #write results for NONHISTORICAL SEGMENT
            print('NONHISTORICAL')
            results_nonhist = make_results(classifier, X_test[nonhist_mask], y_test[nonhist_mask], model, file_name)
            record_results(model, results_nonhist, file_name, 'NONHISTORICAL')
    
    #fit the model
    # print('='*20)
    # print('FITTING MODEL')
    # print('INTERNAL WORKINGS')
    # # classifier.fit(X_train, y_train)
    # coefs = sorted([(i,round(j,3)) for i , j in zip(X_test.columns, classifier.coef_[0])], key = lambda x: abs(x[1]), reverse = True)
    # if args.model =='log':
    #     for i in coefs:
    #         print("{}:\t\t\t{}".format(i[0],i[1]))
    # elif args.model == 'dtree':
    #     print('something to come')

    # #predict the classes
    # print('='*20)
    # print('PREDICTING')
    # y_pred = classifier.predict(X_test)

    # print('GENERAL')
    # print('Accuracy score: ', accuracy_score(y_test, y_pred))
    # print('CONFUSION MATRIX')
    # print(confusion_matrix(y_test, y_pred))  
    # print(classification_report(y_test, y_pred))

    # #look into how each model performs historical and nonhistorical subsets
    # hist_mask = X_test['count_app'] >= 1
    # nonhist_mask = X_test['count_app'] == 0
    # if args.group == 'all':
    #     print('HISTORICAL')
    #     print('Accuracy score: ', accuracy_score(y_test[hist_mask], y_pred[hist_mask]))
    #     print('CONFUSION MATRIX')
    #     print(confusion_matrix(y_test[hist_mask], y_pred[hist_mask]))  
    #     print(classification_report(y_test[hist_mask], y_pred[hist_mask]))

    #     print('NON-HISTORICAL')
    #     print('Accuracy score: ', accuracy_score(y_test[nonhist_mask], y_pred[nonhist_mask]))
    #     print('CONFUSION MATRIX')
    #     print(confusion_matrix(y_test[nonhist_mask], y_pred[nonhist_mask]))  
    #     print(classification_report(y_test[nonhist_mask], y_pred[nonhist_mask]))

    # #print the ROC graph
    # logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
    # fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
    # plt.figure()
    # plt.plot(fpr, tpr, label='SVM (area = %0.2f)' % logit_roc_auc)
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_size', type = float, default = .2,
            help = 'the ratio of test to train in decimal form')
    parser.add_argument('-one_hot', default = False, 
            help = 'specify True to make the categorical variables into one hot vector embeddings')
    parser.add_argument('-group', default = 'all',
            help = 'pick all, historical, or nonhistorical to filter training data')
    parser.add_argument('-no_cancel', default = False, 
            help = 'Choose True to remove cancelled appointmet from dataset')
    parser.add_argument('-sample_type', nargs = '*', default = None, 
            help = 'Fill with the oversampling method and then values to plug into method after word, seperate by spaces')
    args = parser.parse_args()
    # print(parser)
    main(args)


# cross validation
# add prob output into data
# make one hot encoding #ryan done

# design vector of paramters to input into the script
#   inputs: 
#       oversampling techniques
#       historical vs nonhistorical vs all
#       (run all models for each combiantion of parameters)
#       one hot encoding (confirm that the category data type does this on back end)
#       remove cancellations
#       remove cancellations only due to doctor

# design the analysis metrics for each model type
#       All  
#           ROC, confusion matrix, accuracy, precision, recall, AUC
#       Regression
#           MSE
#       Decision
#           information gain metric (gini, entropy)
#           splits
#           gini
#       SVM
#           distance from support vectors
#       Random Forest
#           number of trees

# writing results to some file
    

# keep calm, and keep feature engineering
#       add feature for changes in insurance
#       add feature for median household income based on zip codes

