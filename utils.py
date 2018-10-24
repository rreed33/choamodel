# This file will record more extraneous results of the model/data

import dataprocessing
import sklearn
from sklearn.feature_selection import chi2
import argparse
from statsmodels.stats.outliers_influence import variance_inflation_factor    

def gen_chi2(X, y, one_hot='True'):

	scores, p_values = chi2(X, y)

	results = pd.DataFrame( [scores, p_values], index = X.columns)
	results.to_csv('data_metrics/feature_p_one_hot_{}.csv'.format(one_hot))

	return results


#this function returns the multilinearity of the data with the model
def vif(data, thresh = 5.0):
	variables = list(range(X.shape[1]))
	dropped = True
	while dropped:
		dropped = False
		vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
				for ix in range(X.iloc[:, variables].shape[1])]

		maxloc = vif.index(max(vif))
		if max(vif) > thresh:
			print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
					'\' at index: ' + str(maxloc))
			del variables[maxloc]
			dropped = True

	print('Remaining variables:')
	print(X.columns[variables])
	return X.iloc[:, variables]


def main(args, X, y):

	if args.function:
		result = gen_chi2(X, y, args.one_hot)
	elif agrs.function == 'vif':
		result = vif(X, y)
	else:
		raise "InvalidFunctionError"

	return 

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-function', type = str, 
				help = "enter chi2 or vif")
	parser.add_argument('-one_hot', type = str, default = 'True',
						help = 'set to True to set everything to boolean')
	args = parser.parse_args()

	# load all the relevant data
	df = dataprocessing.main(one_hot = 'True', group = 'historical', no_cancel = 'False',  original = 'False')
	X = df.drop(['No_Show','count'], axis = 1)
	y = df['No_Show']

	main(args, X, y)



