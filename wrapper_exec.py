# this wrapper script will run all the combinations of inputs
# for the model in NN_austin.py

import sys

test_size	= [.2, .5]
one_hot		= [True, False]
groups		= ['all', 'historical', 'nonhistorical']
no_cancel	= [True, False]
sample_type	= ['over_under', 'underTomek', 'underCentroid', 'overSMOTE']

for size in test_size:
	for hot in one_hot:
		for group in groups:
			for can in no_cancel:
				for typ in sample_type:
					os.sytem("python NN_austin.py -test_size {} -one_hot {} -group {} \
								-no_cancel {} -sample_type {}".format(size, hot, group, can, typ))
					os.system("git add .")
					os.system("git commit -m \"This commit: {}\"".format(', '.join([
											size, hot, group, can, typ])))
					os.system("git push origin master")

