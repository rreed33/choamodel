# This file will record more extraneous results of the model/data

import dataprocessing
import sklearn
from sklearn.feature_selection import chi2
import argparse

def gen_chi2(one_hot='True'):
	#import the edited data
	df = dataprocessing.main(one_hot = one_hot, group = 'historical', no_cancel = 'False',  original = 'False')

	X = df.drop(['No_Show','count','Made_Lead_Days'], axis = 1)
	y = df['No_Show']

	scores, p_values = chi2(X, y)

	results = pd.DataFrame( [scores, p_values], index = X.columns)
	results.to_csv('data_metrics/feature_p_one_hot_{}.csv'.format(one_hot))

	return results


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-one_hot', type = str,
						help = 'set to True to set everything to boolean')
	args = parser.parse_args()

	gen_chi2(args.one_hot)
