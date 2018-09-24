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
##%matplotlib inline
# def no_show_sum(df):
#   df[df['No_Show']==0].cumcount()

def distance(lat1, lon1, lat2, lon2):
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

def edit(dataframe):
    #use this function to organize all the edits to the raw data before processing
    print(dataframe.dtypes)
    print(dataframe['No_Show'])

    #get the bird's eye distance to from home to office
    dataframe['distance'] = np.vectorize(distance)(dataframe['Patient_Latitude'], -1*dataframe['Patient_Longitude'],
                                        dataframe['Dept_Location_Latitude'], -1*dataframe['Dept_Location_Longitude'] )

    #first let's see how many appointments each person has had up until then and hwo many they miseed
    dataframe['count_app'] = dataframe.groupby('Sibley_ID', sort = 'Appt_Date').cumcount()
    dataframe['count_miss']= dataframe[dataframe['No_Show']==0].groupby('Sibley_ID', sort = 'Appt_Date').cumcount()
    print(dataframe.groupby('Sibley_ID', sort='Appt_Date').cumcount())


    #calculate bird eye distance 
    return dataframe


def main(args):
    #EVERYTHING ABOVE HERE CAN BE IGNORED
    df = dataprocessing.main(args.group, args.no_cancel)

    df['Payor_Type_ID'].astype(int).astype('category')
    df['Dept_ID'].astype('category')
    df['Provider_ID'].astype('category')
    df['Appt_Logistics_Type_ID'].astype('category')
    df['Visit_Type_ID'].astype('category')

    #print(np.any(np.isnan(df)))
    #print(np.all(np.isfinite(df)))

    #split the data into dependent and predictor
    X = df.drop('No_Show', axis=1)  
    y = df['No_Show']

    # over sampling the no show class to even class distribution
    # RYAN: over_sample will be a list of the inputs you write, indexed according to imput order, first being model method type
    if args.over_sample:
        # args.over_sample[1:] are the parameters to plug into 
        from imblearn.combine import SMOTETomek
        smt = SMOTETomek(ratio='auto')
        X, y = smt.fit_sample(X, y)

        from imblearn.under_sampling import TomekLinks
        tl = TomekLinks(return_indices=True, ratio='majority')
        X, y, id_tl = tl.fit_sample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)

    #build the model
    print('='*20)
    print('INITIALIZING {} MODEL'.format(args.model))
    if args.model == 'SVM':
        classifier = SVC()
    elif args.model == 'log':
        classifier = LogisticRegression()
    elif args.model == 'dtree':
        classifier = DecisionTreeClassifier()
    elif args.model == 'rf':
        classifier = RandomForestClassifier(n_estimators=100)

    model_types = ['SVM', 'log', 'dtree', 'rf', 'kernel_log', 'ridge_log']
    for model in model_types:

        #start a classifier
        if model == 'SVM':
            classifier = SVC()
        elif model == 'log':
            classifier = LogisticRegression()
        elif model == 'dtree':
            classifier = DecisionTreeClassifier()
        elif model == 'rf':
            classifier = RandomForestClassifier(n_estimators=100)
        elif model == 'kernel_log':
            classifier = KernelRidge()
            print('stuff to return')
        elif model == 'ridge_log':
            print('stuff to return')

    #fit the model
    print('='*20)
    print('FITTING MODEL')
    print('INTERNAL WORKINGS')
    classifier.fit(X_train, y_train)
    coefs = sorted([(i,round(j,3)) for i , j in zip(X_test.columns, classifier.coef_[0])], key = lambda x: abs(x[1]), reverse = True)
    if args.model =='log':
        for i in coefs:
            print("{}:\t\t\t{}".format(i[0],i[1]))
    elif args.model == 'dtree':
        print('something to come')

    #predict the classes
    print('='*20)
    print('PREDICTING')
    y_pred = classifier.predict(X_test)

    print('GENERAL')
    print('Accuracy score: ', accuracy_score(y_test, y_pred))
    print('CONFUSION MATRIX')
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred))

    #look into how each model performs historical and nonhistorical subsets
    hist_mask = X_test['count_app'] >= 1
    nonhist_mask = X_test['count_app'] == 0
    if args.group == 'all':
        print('HISTORICAL')
        print('Accuracy score: ', accuracy_score(y_test[hist_mask], y_pred[hist_mask]))
        print('CONFUSION MATRIX')
        print(confusion_matrix(y_test[hist_mask], y_pred[hist_mask]))  
        print(classification_report(y_test[hist_mask], y_pred[hist_mask]))

        print('NON-HISTORICAL')
        print('Accuracy score: ', accuracy_score(y_test[nonhist_mask], y_pred[nonhist_mask]))
        print('CONFUSION MATRIX')
        print(confusion_matrix(y_test[nonhist_mask], y_pred[nonhist_mask]))  
        print(classification_report(y_test[nonhist_mask], y_pred[nonhist_mask]))

    #print the ROC graph
    logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='SVM (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()


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
    parser.add_argument('-over_sample', nargs = '*', default = None, 
            help = 'Fill with the oversampling method and then values to plug into method after word, seperate by spaces')
    args = parser.parse_args()

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

