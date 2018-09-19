import argparse
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC  
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, pairwise
import FukuML
import FukuML.KernelLogisticRegression as kernel_logistic_regression
import math
##%matplotlib inline
# def no_show_sum(df):
# 	df[df['No_Show']==0].cumcount()

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
	# print(dataframe['No_Show'])

	#get the bird's eye distance to from home to office
	dataframe['distance'] = np.vectorize(distance)(dataframe['Patient_Latitude'], -1*dataframe['Patient_Longitude'],
										dataframe['Dept_Location_Latitude'], -1*dataframe['Dept_Location_Longitude'] )

	#first let's see how many appointments each person has had up until then and hwo many they miseed
	dataframe['count_app'] 		= dataframe.groupby('Sibley_ID', sort = 'Appt_Date').cumcount()
	dataframe['count_miss']		= dataframe[dataframe['Appt_Status_ID']==1].groupby('Sibley_ID', sort = 'Appt_Date').cumcount()
	dataframe['count_cancel']	= dataframe[(dataframe['Appt_Status_ID'] != 2 )& (dataframe['Appt_Status_ID'] != 4)].groupby('Sibley_ID', sort = 'Appt_Date').cumcount()
	dataframe['No_Show'] 		= (dataframe['Appt_Status_ID']==4).astype(int)
	# print(dataframe.groupby('Sibley_ID', sort='Appt_Date').cumcount())


	#calculate bird eye distance 
	return dataframe


def main(args):
	df = pd.read_csv("data/choa_real_encounters.csv")
	df_dept = pd.read_csv('data/choa_dept.csv')
	df = df.merge(df_dept, on = 'Dept_ID') 
	print(df.head())
	df = edit(df)

	df['Payor_Type_ID'] = df['Payor_Type_ID'].fillna(0, inplace=True)
	
	#drop all the cancellations to simplify behavior
	# df = df[(df['Appt_Status_ID'] ==4) | (df['Appt_Status_ID'] ==2)] 
	# df = df[df['Appt_Status_ID'] !=2]
	# print(df.dtypes)
	print('Provider_ID' in df.columns)
	df = df.drop([ 'Provider_ID', 'Visit_Type_ID', 'Num_No_Show_Encounters_AllTime', 'Num_Canceled_Encounters_Since', 'Num_No_Show_Encounters_Since', 'Num_Canceled_Encounters_AllTime', 'Appt_Status_ID',
					'Made_Lead_Days_Work','Encounter_ID','Appt_Date','Appt_Time','Appt_Made_Date',
	              'Appt_Made_Time','Sibley_ID', 'Dept_Name', 'Dept_Abbr_3', 'Dept_Abbr_4'
	              ], axis = 1)
	# 'Num_No_Show_Encounters_AllTime', 'Num_Canceled_Encounters_Since', 'Num_No_Show_Encounters_Since', 'Num_Canceled_Encounters_AllTime', 'Appt_Status_ID'



	df.fillna(0, inplace = True)

	# print(df)
	# print(df.shape)

	df['Payor_Type_ID'].astype(str).astype(int).astype('category')
	df['Dept_ID'].astype('category')
	# df['Provider_ID'].astype('category')
	df['Appt_Logistics_Type_ID'].astype('category')
	# df['Visit_Type_ID'].astype('category')
	
	#write a file here to peak into the data
	df.to_csv('data/choa_intermediate.csv')

	print(df.dtypes)

	##print(np.any(np.isnan(df)))
	##print(np.all(np.isfinite(df)))

	#split the train and testing datasets
	X = df.drop('No_Show', axis=1)  
	y = df['No_Show']

	##df = df.reset_index()
	##df = df.dropna()

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=args.train_size, random_state = args.random)
	train, test = train_test_split(df, test_size = .7)

	#kernel trick
	if args.kernel:
		print('='*20)
		print('KERNEL TRICK')
		X_train = pairwise.rbf_kernel(X_train, X_train, gamma=1)
		X_test = pairwise.rbf_kernel(X_test, X_test, gamma=1)
		# X_train = rbf_feature.fit_transform(X_train)
		# X_test = rbf_feature.fit_transform(X_test)

	#build the model
	print('='*20)
	print('INITIALIZING {} MODEL'.format(args.model))
	if args.model == 'SVM':
		classifier = SVC()
	elif args.model == 'log':
		classifier = LogisticRegression(solver='lbfgs')
	elif args.model == 'kernel_log':
		train.to_csv('data/choa_space_train.csv', sep = ' ', header = False)
		classifier = kernel_logistic_regression.KernelLogisticRegression()

	#fit the model
	print('='*20)
	print('FITTING MODEL')
	# X_train['y']= y_train
	classifier.fit(X_train, y_train) if args.model !='kernel_log' else classifier.load_train_data('data/choa_space.csv')

	#print the coefficients of features
	print('='*20)
	print(len(list(X_train.columns)), len(list(classifier.coef_)))
	print(X_train.columns, classifier.coef_)

	# print(sorted([(i,j) for i,j in zip(list(X_train.columns), list(classifier.coef_))], key= lambda x: x[1]))

	#predict the classes
	print('='*20)
	print('PREDICTING')
	X_test.to_csv('data/choa_space_test.csv', sep = ' ', header = False)
	y_pred = classifier.predict(X_test) if args.model != 'kernel_log' else classifier.load_test_data('data/choa_space_test.csv')
	y_train_pred = classifier.predict(X_train)
	# classifier.train()

	acc = accuracy_score(y_test, y_pred)
	acc_train = accuracy_score(y_train, y_train_pred)
	print('Accuracy test score: ', acc)
	print('Accuracy train score: ', acc_train)
	print('CONFUSION MATRIX for {} of Data'.format(1-args.train_size))
	print(confusion_matrix(y_test, y_pred))  
	print(classification_report(y_test, y_pred))

	logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
	fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
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

	return acc, acc_train

# def record(args, results):
# 	if args.


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-model')
	parser.add_argument('-kernel', default= None)
	parser.add_argument('-train_size', default = .2, type = float)
	parser.add_argument('-random', default = 1001, type = int)
	args = parser.parse_args()

	acc_test_lst = []
	acc_train_lst = []
	for i in range(1, 10):
		args.train_size =  i*.1
		acc_test, acc_train = main(args)
		acc_test_lst.append(acc_test)
		acc_train_lst.append(acc_train)
	# print( [ (j*.1, i) for i,j in zip(acc_lst, range(1,10))])

	if args.model == 'log':
		model = 'Logistic Regression'
	else:
		model = 'Some'

	# plt.clear()
	x = [i*.1 for i in range(1,10)]
	plt.title('{} Accuracy Curve'.format(model))
	plt.ylabel('Accuracy')
	plt.ylim([.8,1])
	plt.xlabel('Train Size Proportion')
	plt.plot(x, acc_test_lst, label = 'Testing Acc')
	plt.plot(x, acc_train_lst,label = 'Training Acc')
	plt.legend()

	max_test = np.argmax(acc_test_lst)
	max_train= np.argmax(acc_train_lst)

	plt.scatter([x[max_test], x[max_train]], [acc_test_lst[max_test], acc_train_lst[max_train]])
	plt.annotate(acc_test_lst[max_test], (x[max_test], acc_test_lst[max_test]))
	plt.annotate(acc_train_lst[max_train], (x[max_train], acc_test_lst[max_train]))

	plt.show()

#TO DO:
# confirm categorical feature behavior in log reg model
# look into log reg theory
# figure out significance some how?
#  find ou twhy the acc diff after changing the data
# fiure out the kernel for log reg
# make more visualizations