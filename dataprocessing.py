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


	#get the bird's eye distance to from home to office
	dataframe['distance'] = np.vectorize(distance)(dataframe['Patient_Latitude'], -1*dataframe['Patient_Longitude'],
						 				dataframe['Dept_Location_Latitude'], -1*dataframe['Dept_Location_Longitude'] )

	#first let's see how many appointments each person has had up until then and hwo many they miseed
	dataframe['No_Show']		= (dataframe['Appt_Status_ID']==4).astype(int)
	dataframe['Cancelled']		= ((dataframe['Appt_Status_ID']!=4)&(dataframe['Appt_Status_ID']!=2)).astype(int)

	grouped_sibley				= dataframe.groupby('Sibley_ID', sort = 'Appt_Date')
	dataframe['count_app'] 		= grouped_sibley.cumcount()
	dataframe['count_miss']		= grouped_sibley['No_Show'].cumsum()
	dataframe['count_cancel']	= grouped_sibley['Cancelled'].cumsum()
	# dataframe['diff_pay_count']	= grouped_sibley['Payor_Type_ID'].agg( lambda x: sum(x.diff() != 0) )
	
	# earliest_date				= 
	dataframe['Appt_Date'] 		= pd.to_datetime(dataframe['Appt_Date'])
	

	dataframe = dataframe.drop(['Encounter_ID','Appt_Date','Appt_Time','Appt_Made_Date',
                  'Appt_Made_Time', 'Dept_Name', 'Dept_Abbr_3', 'Dept_Abbr_4'], axis=1)

	dataframe.fillna(0, inplace = True)

	return dataframe


def main(group='all', no_cancel = False, one_hot = False):
	df = pd.read_csv("../data/ENCOUNTERS_RAW.csv")
	df_dept = pd.read_csv('../data/DEPT_RAW.csv')
	df = df.merge(df_dept, on = 'Dept_ID') 
	df.sort_values(by = 'Appt_Date', inplace = True)
	df = edit(df)

	#only look at those with history
	# df = df[df['count_app'] == 1]

	#determine one hot encoding for variables:
	if one_hot:
		# use pd.concat to join the new columns with your original dataframe
		df = pd.concat([df,pd.get_dummies(df['Dept_ID'], prefix='dept')],axis=1)
		df = pd.concat([df,pd.get_dummies(df['Provider_ID'], prefix='provider')],axis=1)
		df = pd.concat([df,pd.get_dummies(df['Appt_Logistics_Type_ID'], prefix='appt_log_type')],axis=1)
		df = pd.concat([df,pd.get_dummies(df['Visit_Type_ID'], prefix='visit_type')],axis=1)
		df = pd.concat([df,pd.get_dummies(df['Patient_Age_Bucket_ID'], prefix='age_bucket')],axis=1)
		df = pd.concat([df,pd.get_dummies(df['Payor_Type_ID'], prefix='payor_type')],axis=1)

		# now drop the original 'country' column (you don't need it anymore)
		df.drop(['Dept_ID'],axis=1, inplace=True)
		df.drop(['Provider_ID'],axis=1, inplace=True)
		df.drop(['Appt_Logistics_Type_ID'],axis=1, inplace=True)
		df.drop(['Visit_Type_ID'],axis=1, inplace=True)
		df.drop(['Patient_Age_Bucket_ID'],axis=1, inplace=True)
		df.drop(['Payor_Type_ID'],axis=1, inplace=True)


	#divide the group into historical and nonhistorical patients
	if group == 'historical':
		df = df[df['count_app'] >= 1]
	elif group == 'nonhistorical':
		df = df[df['count_app'] == 0]

	#get rid of cancellations
	if no_cancel:
		df = df[(df['Appt_Status_ID'] == 4) | (df['Appt_Status_ID'] == 2)]

	#df[['Payor_Type_ID']] = df[['Payor_Type_ID']].fillna(value=0)
	df = df.drop([ 'Num_Canceled_Encounters_Since',
                       'Num_No_Show_Encounters_Since',
                       'Num_Canceled_Encounters_AllTime',
                       'Num_No_Show_Encounters_AllTime',
                       'Appt_Status_ID'],
                         axis = 1)
                    
	df.to_csv('../data/choa_intermediate.csv')
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


#To Do 9/24 8pm
#  Ryan
# focus on oversampling
#  encoding for date variables
#  fix sort
#
#  Austin
#  file wriitng, coompresive stats, wirte model stats