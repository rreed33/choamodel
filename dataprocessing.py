import argparse
import sklearn
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
import math

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
	dataframe['count_miss']		= dataframe[dataframe['Appt_Status_ID']==4].groupby('Sibley_ID', sort = 'Appt_Date').cumcount()
	dataframe['count_cancel']	= dataframe[(dataframe['Appt_Status_ID'] != 2 )& (dataframe['Appt_Status_ID'] != 4)].groupby('Sibley_ID', sort = 'Appt_Date').cumcount()
	dataframe['No_Show'] 		= (dataframe['Appt_Status_ID']==4).astype(int)
	# print(dataframe.groupby('Sibley_ID', sort='Appt_Date').cumcount())


	#calculate bird eye distance 
	return dataframe


def main(group='all', no_cancel = False):
	df = pd.read_csv("../data/ENCOUNTERS_RAW.csv")
	df_dept = pd.read_csv('../data/DEPT_RAW.csv')
	df = df.merge(df_dept, on = 'Dept_ID') 
	df = edit(df)

	#only look at those with history
	# df = df[df['count_app'] == 1]

	#divide the group into historical and nonhistorical patients
	if group == 'historical':
		df = df[df['count_app'] > 1]
	elif group == 'nonhistorical':
		df = df[df['count_app'] == 1]

	#get rid of cancellations
	if no_cancel:
		df = df[(df['Appt_Status_ID'] == 4) | (df['Appt_Status_ID'] == 2)]

	df[['Payor_Type_ID']] = df[['Payor_Type_ID']].fillna(value=0)
	df = df.drop([ 'Num_Canceled_Encounters_Since',
                       'Num_No_Show_Encounters_Since',
                       'Num_Canceled_Encounters_AllTime',
                       'Num_No_Show_Encounters_AllTime',
                       'Appt_Status_ID'],
                         axis = 1)
                    
	df.to_csv('../choa_intermediate.csv')
	return df

if __name__ == '__main__':
	main()


#TO DO:
# confirm categorical feature behavior in log reg model
# look into log reg theory
# figure out significance some how?
#  find ou twhy the acc diff after changing the data
# fiure out the kernel for log reg
# make more visualizations
