from itertools import combinations

def main(numSlots, numOverbook, coverageMorning, rateMorning):
	#Slots outside of high confidence slots, which we care about
	morningSlots = int(coverageMorning*numSlots)

	# remainingSlots = int(morningSlots - numOverbook)
	expected_noShows = int(rateMorning*morningSlots)  #number of no shows in morningSlots
	slots = list(range(1, morningSlots + 1))
	comb = combinations(slots, expected_noShows)
	print("Total Slots: ", numSlots)
	print("Overbooking Slots: ", numOverbook)
	print("Low confidence coverage: ", coverageMorning)
	print("No-show rate of low confidence: ", rateMorning)
	print("Morning Slots: ", morningSlots)
	# print("Remaining Slots: ", remainingSlots)
	print("Expected No Shows: ", expected_noShows)
	print("Slots: ", slots)
	print(list(comb))
	return list(comb)


def avg_queue(s, o, l1, l2):
	# s: (int) the number of slots
	# o: (int) the number of overbooking slots
	# l1:(int) the index of the earliest no-show slot
	# l2:(int) the index of the latest no-show slot
	return ( o*(s-2) -2*s + l1 + l2 + 1)/ (s+o-2)

if __name__ == '__main__':
	main(16, 2, .75, 1/6)