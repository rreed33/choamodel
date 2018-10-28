import pandas as pd 
import numpy as np 
import json
import csv
import argparse
import time
import math

from uber_rides.session import Session
from uber_rides.client import UberRidesClient

session = Session(server_token="0-S2sv9ugj9ALa7nofY8gW5k7B_YkgBv9dcKGBMO")
client = UberRidesClient(session)

def main(args):
	source = pd.read_csv( '../data/ENCOUNTERS_RAW.csv')
	source_end = pd.read_csv('../data/choa_dept.csv')

	# source.drop_duplicates(subset=['Encounter_ID', 'Patient_Latitude', 'Patient_Longitude',"Dept_ID"], inplace = True)
	patient = source[['Encounter_ID', 'Patient_Latitude', 'Patient_Longitude',"Dept_ID"]]

	dept = source_end[['Dept_ID', 'Dept_Location_Latitude', 'Dept_Location_Longitude']]
	df = patient.merge(dept)

	def distance(lat1, lon1, lat2, lon2):
	    radius = 6371 # km
	    # lon1 = -1*lon1
	    # lon2 = -1*lon2

	    dlat = math.radians(lat2-lat1)
	    dlon = math.radians(lon2-lon1)
	    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
	        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
	    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	    d = radius * c

	    return d
	df['Patient_Longitude'] = df['Patient_Longitude'] * -1
	df['Dept_Location_Longitude'] = df['Dept_Location_Longitude']*-1
	df['distance_bird'] = np.vectorize(distance)(df['Patient_Latitude'], df['Patient_Longitude'],
						 				df['Dept_Location_Latitude'], df['Dept_Location_Longitude'] )
	
	print df
	# print(patient)
	# print(dept)
	# print("Dept ID Type")
	# print(type(patient["Dept_ID"].iloc[0]))
	# csvfile = open(args.out_file,"wb")
	# fieldnames = ["Encounter_ID","Uber_Price"]
	# writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
	# writer.writeheader()

	def uber(pat_lat, pat_long, dept_lat, dept_long):
		# try:
		pat_long = pat_long
		dept_long = dept_long
		print((pat_lat, pat_long, dept_lat, dept_long))
		to_response = client.get_price_estimates(
						start_latitude=pat_lat,
						start_longitude=pat_long,
						end_latitude=dept_lat,
						end_longitude=dept_long,
						seat_count=2)
		to_estimate = to_response.json.get('prices')
		print to_estimate
		to_price = (float(to_estimate[0]["low_estimate"]) + float(to_estimate[0]["high_estimate"]))/2 if type(to_estimate) == list else (float(to_estimate["low_estimate"]) + float(to_estimate["high_estimate"]))/2
		

		from_response = client.get_price_estimates(
			start_latitude=dept_lat,
			start_longitude=dept_long,
			end_latitude=pat_lat,
			end_longitude=pat_long,
			seat_count=2)


		from_estimate = from_response.json.get('prices')[0]
		print from_estimate
		from_price = (float(from_estimate[0]["low_estimate"]) + float(from_estimate[0]["high_estimate"]))/2
		from_price = (float(from_estimate[0]["low_estimate"]) + float(from_estimate[0]["high_estimate"]))/2 if type(from_estimate) ==list else (float(from_estimate["low_estimate"]) + float(from_estimate["high_estimate"]))/2
		

		total_price = (to_price) + (from_price)
		print total_price
		return total_price
		# except:
		# 	return 0

	df = df[df['distance_bird'] < 100]
	# df['uber_round_price'] =  np.vectorize(uber)(df['Patient_Latitude'], df['Patient_Longitude'],
	# 					 				df['Dept_Location_Latitude'], df['Dept_Location_Longitude'] )
	# df['uber_round_price'] = df[['Patient_Latitude', 'Patient_Longitude','Dept_Location_Latitude', 'Dept_Location_Longitude']].apply(lambda x: uber(x[0], x[1],x[2],x[3]),axis = 0)
	lst = []
	for i in range(100):
		print i
		lst.append(uber(df.iloc[i]['Patient_Latitude'], df.iloc[i]['Patient_Longitude'],df.iloc[i]['Dept_Location_Latitude'],df.iloc[i]['Dept_Location_Longitude']))
	print df
	df.to_csv('uber.csv')
	return
	"""
	for k in range(len(patient)):
		for l in range(len(dept)):
			if patient["Dept_ID"].iloc[k] == 14 and dept["Dept_ID"].iloc[l] == 14:
				try:
					patient_lat = float(patient['Patient_Latitude'].iloc[k])
					patient_long = float(-1*patient['Patient_Longitude'].iloc[k])

					dept_lat = float(dept['Dept_Location_Latitude'].iloc[l])
					dept_long = float(-1*dept['Dept_Location_Longitude'].iloc[l])

					to_response = client.get_price_estimates(
						start_latitude=patient_lat,
						start_longitude=patient_long,
						end_latitude=dept_lat,
						end_longitude=dept_long,
						seat_count=2)
					
					to_estimate = to_response.json.get('prices')
					to_price = (float(to_estimate[0]["low_estimate"]) + float(to_estimate[0]["high_estimate"]))/2
					

					from_response = client.get_price_estimates(
						start_latitude=dept_lat,
						start_longitude=dept_long,
						end_latitude=patient_lat,
						end_longitude=patient_long,
						seat_count=2)

					from_estimate = from_response.json.get('prices')
					from_price = (float(from_estimate[0]["low_estimate"]) + float(from_estimate[0]["high_estimate"]))/2
					

					total_price = (to_price) + (from_price)
					print("total_estimate:"+str(total_price))

					csvdict = {"Encounter_ID":str(patient["Encounter_ID"].iloc[k]), "Uber_Price":total_price}
					print(csvdict)

					writer.writerow(csvdict)
					time.sleep(0.5)
				except:
					pass
			else:
				pass
		"""

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-in_file', type = str, default = '../data/check.csv',
		help = 'provide the excel file name including file handle')
	parser.add_argument('-out_file', type = str, default = 'uber.csv',
		help = 'provide the filename of the csv file to print to (eg choa1.csv)')
	args = parser.parse_args()

	main(args)







