# this script will return the average queue time given the
# placement of no shows in the variable section of the choa schedule
from __future__ import division

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import argparse
from itertools import combinations

def queue_time( s, k, l, a):
# inputs
# int: location of the first no show
# int: location of the second no show

# output
# float: average number of appointments waited
	# fun2 = lambda s, k, x: k*((s-x)/(s+k-x))
	# fun1 = lambda num_slots, num_over, num_no: (sum([2*i for i in range(1,num_over-1)]) + (num_slots-num_no-num_over+1)*num_over ) / (num_slots + num_over - num_no)

	# normal = lambda s, k, x: k*((s-x)/(s+k-x)) 
	# one = lambda s, k, l1: ( k*(s-1) - s + l1)/ (s+k-1)
	# two = lambda s, k, l1, l2: ( k*(s-2) -2*s + l1 + l2 + 1)/ (s+k-2)

	# there are two parts:
	# part 1: the area of overbooking until slots k -1 
	# part 2: the area after overbooking plus the last overbooking slot
	l    = sorted(l)
	l_ob = [ i if i <= k-1 else 0 for i in l ]
	l_p  = [ i if i >  k-1 else 0 for i in l ]
	x    = len(l)
	x_ob = sum([ 1 if i <= k-1 and i > 0 else 0 for i in l])		# the number of no shows in part 1
	x_p  = sum([ 1 if i >  k-1 and i > 0 else 0 for i in l])		# the number of no shows in part 2
	# print('l:\t{}\nl_ob:\t{}\nl_p:\t{}\nx:\t{}\nx_ob:\t{}\nx_p:\t{}'.format(l, l_ob, l_p, x, x_ob, x_p))

	if l[0] == [1]:
		l_ob = (0,0)
		l_p  = (0,0)
		x    = 0
		x_ob = 0
		x_p  = 0
		s    = s - 1
		k    = k - 1

	# print('\n')
	all_p1 = k*(k-1)
	# print('all_p1;\t', all_p1)
	one_p1 = -1*l_ob[0] - (2*(k-l_ob[0])-1) if l_ob[0] <= k-1 and l_ob[0] > 0 else 0
	# print('one_p1:\t', one_p1)
	two_p1 = 1 - l_ob[1] -(2*(k-l_ob[1])-1) if l_ob[1] <= k-1 and l_ob[1] > 0 else 0
	# print('two_p1:\t', two_p1)
	all_p2 = (s - x_p - k + a + 1)*(k - x_ob)
	# print('all_p2:\t', all_p2)
	btw_12 = -1*(l_p[1] - l_p[0] - 1) if (l_p[0] > k-1 and l_p[1] > k-1) else 0
	# print('btw_12:\t', btw_12)
	aft_fn = -1*x_p*(s - l_p[-1] + a) 
	# print('aft_fn:\t', aft_fn)
	# print('\n')
	

	result = (all_p1 + one_p1 + two_p1 + all_p2 + btw_12 + aft_fn)/(args.s + args.k - 2)
	# print(l, '\t', 'result:\t', result)
	return result

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument('-session_id', type = int, default = 1)
	parser.add_argument('-s', type = int)
	parser.add_argument('-k', type = int)
	parser.add_argument('-l', nargs = '*')
	parser.add_argument('-a', type = int)
	args = parser.parse_args()

	# l = tuple([ int(i) for i in args.l])
	# queue_time( args.s, args.k, l )

	slots = list(range(1, args.s + 1))
	comb = list(combinations(slots, 2))
	for i in range(1, args.s + 1):
		comb.append((i,i))

	overBookSlots = list(range(1, args.s + 1))
	avgLengths = []

	for i in overBookSlots:
		totalTime = 0
		for pair in comb:
			totalTime += queue_time(args.s, i, pair, args.a)
		avgQueueTime = totalTime/(len(comb) + args.a)
		avgLengths.append(avgQueueTime)
		print("Overbook slots: ", i, " Average Queueing Time: ", avgQueueTime)


	# totalTime = 0
	# for pair in comb:
	# 	totalTime += queue_time(args.s, args.k, pair)
	# avgQueueTime = totalTime/len(comb)
	# avgLengths.append(avgQueueTime)

	plt.plot(overBookSlots, avgLengths)		
	plt.show()

	print(overBookSlots)
	# print("Average queueing time: ", avgQueueTime)
"""

	# fun_1 = lambda s, k, l1: ( k*(s-1) - s + l1 )/ (s+k-1)
	# fun_2 = lambda s, k, l1, l2: ( k*(s-2) -2*s + l1 + l2 + 1)/ (s+k-2)
	# fun = lambda num_slots, num_over, l1, l2: ( k*(num_slots-2) -2*s + l1 + l2 + 1)/ (s+k-2)
	return fun1(num)


def avg_queue(s, o, l1, l2):
	# s: (int) the number of slots
	# o: (int) the number of overbooking slots
	# l1:(int) the index of the earliest no-show slot
	# l2:(int) the index of the latest no-show slot
	return ( o*(s-2) -2*s + l1 + l2 + 1)/ (s+o-2)

lst_lst = []
lst_x = []
for i in range(5):
	print()
	lst = []
	x = []
	print('overbooking', i)
	# for j in range(i+1, 16):
	# 	x.append(j)
	lst.append(queue_time(16, i, 1, 15, 1))
	print('some', queue_time(16, i, 1, 15, 1))
lst_lst.append(lst)
# lst_x.append(x)

# for i, j in zip(lst_x, lst_lst):
for i in lst_lst:
	plt.plot( i)
	plt.ylim(0,3)
	plt.show()
"""
