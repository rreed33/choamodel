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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, make_scorer, accuracy_score, precision_score, recall_score, f1_score
import math
import imblearn
import os
import datetime as dt 
from sklearn import tree

import dataprocessing
from gen_results import main as make_results

from sklearn.model_selection import StratifiedKFold, cross_validate
##K-folds with equal class distribtions

# def strBool(string):


def record_file(args, df):
    parameter_names = ['-test_size', '-one_hot', '-group', '-no_cancel', '-sample_type', '-original', '-cv', '-clusters']
    parameters      = [args.test_size, args.one_hot, args.group, args.no_cancel, args.sample_type, args.original, args.cv, args.clusters]
    file_name   = '../choamodel/results/'
    file_ext     = '_'.join([str(i)+'_'+str(j) for i,j in zip(parameter_names, parameters)])
    file_name = file_name + file_ext + '/'

    print(file_name)
    if not os.path.exists(file_name):
        print('made')
        os.makedirs(file_name)

    with open(file_name+'results.txt', 'w') as f:
        f.write('---This file was generate as a part of Senior Design by team 14 on {} ---\n'.format(dt.datetime.today()))
        f.write('\nCOMMAND LINE INPUT: \n' +'python NN_austin.py '+', '.join([str(i)+': '+str(j) for i, j in zip(parameter_names, parameters)]))
        f.write('\nTraining Group:\t{}\nNumber of Encounters: {}\nNumber of Patients: {}\nNumber of Features: {}'.format(args.group.upper(),len(df), len(np.unique(df['Sibley_ID'])), len(df.columns)))
        f.write('\nNumber of No Shows:\t{}\n'.format(sum(df['No_Show'])))
        f.write('\nFeature Names:\n{}\n'.format(', '.join([i for i in df.keys()])))
    return file_name


def record_results(model, results, file_name, group):

    with open(file_name+'results.txt', 'a') as f:
        f.write('\n\n----------Writing Results for __{}_{}__ ----------\n'.format(model, group))
        f.write('Accuracy Score:\t{}\n'.format(results['Accuracy Score']))
        f.write('Confusion Matrix:\n{}\n'.format(results['Confusion Matrix']))
        f.write('Classification Report:\n{}\n'.format(results['Classification Report']))
        f.write('ROC AUC Score:\t{}\n'.format(results['ROC AUC']))
        if model =='log':
            f.write('Coefficients:\n{}'.format(results['coef']))


def write_pred(file_name, model, X_train, y_test, y_pred):
    #this function writes the prediciton with the whole predictors as well
    file_name = '../data/predictions/'+file_name
    test_results = X_train.copy()
    if not os.path.exists(file_name):
        print('made')
        os.makedirs(file_name)

    test_results['No_Show'] = y_test
    test_results['predict'] = y_pred
    test_results.to_csv(file_name+'/'+model+'_test_pred.csv')



def main(args):
    #EVERYTHING ABOVE HERE CAN BE IGNORED
    if args.generate_data == 'True':
        df = dataprocessing.main(args.group, args.no_cancel, args.one_hot, args.original, args.clusters)
    elif args.generate_data == 'False':
        df = pd.read_csv('../data/choa_group_{}_no_cancel_{}_one_hot_{}_original_{}_intermediate.csv'.format(
                args.group, args.no_cancel, args.one_hot, args.original))
    else:
        print("ERROR: CORRECT 'generate_data' ARGUMENT TO 'True' or 'False'" )

    #split the data into dependent and predictor
    X = df.drop(['No_Show','Sibley_ID', 'count'], axis=1)  
    y = df['No_Show']
    filter_ = df['count']
    print(X.dtypes)

    #record initial metrics about the dataset
    file_name = record_file(args, df)

    X_train, X_test, y_train, y_test, filter_train, filter_test = train_test_split(X, y, filter_, test_size=args.test_size, random_state = 1001)


    # over sampling the no show class to even class distribution
    # RYAN: over_sample will be a list of the inputs you write, indexed according to imput order, first being model method type
    if args.sample_type == 'overunder':
        # args.over_sample[1:] are the parameters to plug into 
        from imblearn.combine import SMOTETomek
        smt = SMOTETomek(ratio='auto')
        X_train, y_train = smt.fit_sample(X_train, y_train)

    elif args.sample_type == 'underTomek':
        from imblearn.under_sampling import TomekLinks
        tl = TomekLinks(return_indices=True, ratio='majority')
        X_train, y_train = tl.fit_sample(X_train, y_train)

    elif args.sample_type == 'underCentroid':
        from imblearn.under_sampling import ClusterCentroids
        cc = ClusterCentroids(ratio={0: 10})
        X_train, y_train = cc.fit_sample(X_train, y_train)

    elif args.sample_type == 'overSMOTE':
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(ratio='minority')
        X_train, y_train = smote.fit_sample(X_train, y_train)

    

    #build the model
    print('='*20)
    print('INITIALIZING MODELS')
    # , 'kernel_ridge' causes a problem with memory for rn
    # , 'ridge_reg', 'lasso_reg' causes a  problem with binary and continuous target space
    # , 'knn' but id takes so long

    model_types = ['log', 'dtree', 'rf', 'logL1', 'SVM']
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
        elif model == 'logL1':
            print('-'*10)
            print('initializing {} model'.format(model))
            classifier = LogisticRegression(penalty='l1')
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
            classifier = DecisionTreeClassifier()  
            classifier.fit(X_train, y_train) 
            tree.export_graphviz(classifier, out_file='tree.dot',
                     feature_names = X.columns)

        elif model == 'rf':
            print('-'*10)
            print('initializing {} model'.format(model))
            classifier = RandomForestClassifier(n_estimators=100)
            print('-'*10)
            print('fitting {} model'.format(model))
            classifier.fit(X_train, y_train)
        elif model == 'svm':
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
        elif model == 'lasso':
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
        elif model == 'nn':
            print('-'*10)
            print('initializing {} model'.format(model))
            classifier = MLPClassifier(solver='sgd', alpha=1e-10, hidden_layer_sizes=(5, 3), random_state=1, activation = 'tanh')
            print('-'*10)
            print('fitting {} model'.format(model))
            classifier.fit(X_train, y_train)
        elif model == 'logL1':
            print('-'*10)
            print('initializing {} model'.format(model))
            classifier = LogisticRegression(penalty='l1')
            print('-'*10)
            print('fitting {} model'.format(model))
            classifier.fit(X_train, y_train)


        print('GENERAL')
        results = make_results(classifier, X_test, y_test, model, file_name, args.group.upper())
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        kfold = StratifiedKFold(n_splits=10, random_state=42)
        scores = cross_validate(estimator=classifier,
                                          X=X,
                                          y=y,
                                          cv=kfold,
                                          scoring=scoring)
        print("Scores ", scores)
        record_results(model, results, file_name, args.group.upper())
        write_pred(file_name.split('/')[-2], model, X_test, y_test, results['Predictions'])


        if args.group == 'all':
            #split the ALL segment into HISTORICAL AND NONHISTORICAL
            hist_mask = (filter_test > 1).values
            nonhist_mask = filter_test == 1

            #write resutls for HISTORICAL SEGMENT
            print('HISTORICAL')
            results_hist = make_results(classifier, X_test[hist_mask], y_test[hist_mask], model, file_name, args.group)
            record_results(model, results_hist, file_name, 'HISTORICAL')

            #write results for NONHISTORICAL SEGMENT
            print('NONHISTORICAL')
            results_nonhist = make_results(classifier, X_test[nonhist_mask], y_test[nonhist_mask], model, file_name, args.group)
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
    parser.add_argument('-one_hot', type = str, default = 'False', 
            help = 'specify True to make the categorical variables into one hot vector embeddings')
    parser.add_argument('-group', default = 'all',
            help = 'pick all, historical, or nonhistorical to filter training data')
    parser.add_argument('-no_cancel', type = str, default = 'False', 
            help = 'Choose True to remove cancelled appointmet from dataset')
    parser.add_argument('-sample_type', nargs = '*', default = None, 
            help = 'Fill with the oversampling method and then values to plug into method after word, seperate by spaces')
    parser.add_argument('-original', type = str, default = 'False',
            help = 'Set this as True to reset features to original values or special to only have engineered features')
    parser.add_argument('-generate_data', default = 'True', 
            help = 'Generate data from scratch or read from choa_intermediate.csv')
    parser.add_argument('-office', type = str, default = 'all',
            help = 'Type in the name of the office as present in the data')
    parser.add_argument('-cv', type = int, default = 0,
            help = 'Enter an int > 0 to run Stratified k-fold cross validation')
    parser.add_argument('-clusters', type = int, default = 0,
            help = 'Enter an int > 0 to run K-means clustering')
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

