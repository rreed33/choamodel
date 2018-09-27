# this wrapper script will run all the combinations of inputs
# for the model in NN_austin.py

import os

test_size	= [.2, .5]
one_hot		= ['False', 'True']
groups		= ['all', 'historical', 'nonhistorical']
no_cancel	= ['True', 'False']
sample_type	= ['over_under', 'underTomek', 'underCentroid', 'overSMOTE']

size		= .2
count = 0

for hot in one_hot:
	for group in groups:
		for can in no_cancel:
			for typ in sample_type:
				print('Iteration '+str(count))
				print('RUNNING {}'.format("python NN_austin.py -test_size {} -one_hot {} -group {} \
							-no_cancel {} -sample_type {}".format(size, hot, group, can, typ)))
				os.system("python NN_austin.py -test_size {} -one_hot {} -group {} \
							-no_cancel {} -sample_type {}".format(size, hot, group, can, typ))
				os.system("git add .")
				os.system("git commit -m \"This commit: {}\"".format(', '.join([
										str(size), str(hot), str(group), str(can), str(typ)])))
				os.system("git push origin master")
				print('\n'*5)
				count += 1

