# this wrapper script will run all the combinations of inputs
# for the model in NN_austin.py

import os

test_size	= [.2]
one_hot		= [False]
groups		= ['all', 'historical', 'nonhistorical']
no_cancel	= [True, False]
sample_type	= ['none']

for size in test_size:
	for hot in one_hot:
		for group in groups:
			for can in no_cancel:
				for typ in sample_type:
					os.system("git reset --hard HEAD")
					os.system("git pull origin master")
					print('RUNNING {}'.format("python NN_austin.py -test_size {} -one_hot {} -group {} \
								-no_cancel {} -sample_type {}".format(size, hot, group, can, typ)))
					os.system("python NN_austin.py -test_size {} -one_hot {} -group {} \
								-no_cancel {} -sample_type {}".format(size, hot, group, can, typ))
					os.system("git add .")
					os.system("git commit -m \"This commit: {}\"".format(', '.join([
											str(size), str(hot), str(group), str(can), str(typ)])))
					os.system("git push origin master")
					print('\n'*5)

