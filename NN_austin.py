import argparse
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC  
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
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
    df = pd.read_csv("Encounters.csv")
    df_dept = pd.read_csv('choa_dept.csv')
    df = df.merge(df_dept) 
    print(df.head())
    df = edit(df)

    df['Payor_Type_ID'] = df['Payor_Type_ID'].fillna(0, inplace=True)
    df = df.drop(['Encounter_ID','Appt_Date','Appt_Time','Appt_Made_Date',
                  'Appt_Made_Time','Sibley_ID', 'Dept_Name', 'Dept_Abbr_3', 'Dept_Abbr_4'], axis = 1)
    print(df)
    print(df.shape)

    df.fillna(0, inplace = True)

    df['Payor_Type_ID'].astype(str).astype(int).astype('category')
    df['Dept_ID'].astype('category')
    df['Provider_ID'].astype('category')
    df['Appt_Logistics_Type_ID'].astype('category')
    df['Visit_Type_ID'].astype('category')
    

    print(df.dtypes)

    ##print(np.any(np.isnan(df)))
    ##print(np.all(np.isfinite(df)))

    X = df.drop('No_Show', axis=1)  
    y = df['No_Show']

    ##df = df.reset_index()
    ##df = df.dropna()
    from imblearn.combine import SMOTETomek
    smt = SMOTETomek(ratio='auto')
    X, y = smt.fit_sample(X, y)
##  from imblearn.under_sampling import TomekLinks
##  tl = TomekLinks(return_indices=True, ratio='majority')
##  X, y, id_tl = tl.fit_sample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

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

    #fit the model
    print('='*20)
    print('FITTING MODEL')
    classifier.fit(X_train, y_train) 

    #predict the classes
    print('='*20)
    print('PREDICTING')
    y_pred = classifier.predict(X_test)

    print('Accuracy score: ', accuracy_score(y_test, y_pred))
    print('CONFUSION MATRIX')
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred))

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
    parser.add_argument('-model')
    args = parser.parse_args()
    args.model = str('rf')
    main(args)
