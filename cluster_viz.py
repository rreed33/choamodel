#this file will reduce the dataset into 2 dimensions and represent a visual

import dataprocessing
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main(args):

	df = dataprocessing.main()
	reduce_data = TSNE(n_components = 2).fit_transform(df)
	plt.plot(reduce_data)
	plt.savefig('results/clusters.jpg')
	plt.show()
	
	return 





	return

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
	parser.add_argument('-sample_type', default = None, 
			help = 'Fill with the oversampling method and then values to plug into method after word, seperate by spaces')
	parser.add_argument('-original', type = str, default = 'False',
			help = 'Set this as True to reselt features to original values')
	args = parser.parse_args()

	main(args)