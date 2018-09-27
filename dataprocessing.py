import argparse
import sklearn
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
import math
import sklearn.preprocessing

def distance(lat1, lon1, lat2, lon2):
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

def google_distance(df):

	#read in the file with all the distances and times for each encounter and scale 
	dist_df = pd.read_csv('../data/google_dist.csv')
	scaler = sklearn.preprocessing.MinMaxScaler()
	dist_df['Distance'] = scaler.fit_transform(dist_df['Distance'].values.reshape(-1,1))
	dist_df['Duration'] = scaler.fit_transform(dist_df['Duration'].values.reshape(-1,1))

	#join the two dataframes on 'Encounter_ID'
	df = df.merge(dist_df, how = 'left', on='Encounter_ID')

	return df

def edit(dataframe):
	#use this function to organize all the edits to the raw data before processing


	#get the bird's eye distance to from home to office
	# dataframe['distance'] = np.vectorize(distance)(dataframe['Patient_Latitude'], -1*dataframe['Patient_Longitude'],
						 				# dataframe['Dept_Location_Latitude'], -1*dataframe['Dept_Location_Longitude'] )

	#first let's see how many appointments each person has had up until then and hwo many they miseed
	dataframe['No_Show']		= (dataframe['Appt_Status_ID']==4).astype(int)
	dataframe['Cancelled']		= ((dataframe['Appt_Status_ID']!=4)&(dataframe['Appt_Status_ID']!=2)).astype(int)
	
	dataframe['Appt_Date'] 		= pd.to_datetime(dataframe['Appt_Date'])
	#grouped_sibley				= dataframe.groupby('Sibley_ID', sort = 'Appt_Date')
	grouped_sibley				= dataframe.sort_values(['Appt_Date'])
	grouped_sibley				= grouped_sibley.groupby("Sibley_ID")
	dataframe['count_app'] 		= grouped_sibley.cumcount()
	dataframe['count_miss']		= grouped_sibley['No_Show'].transform( lambda x: x.shift().fillna(0).cumsum())
	dataframe['count_cancel']	= grouped_sibley['Cancelled'].transform( lambda x: x.shift().fillna(0).cumsum())
	dataframe['diff_pay_count']	= grouped_sibley['Payor_Type_ID'].apply( lambda x: ((x - x.shift()) != 0).cumsum() )
	
	# Appt_Made_Time into its of year, month date
	dataframe['Appt_Date'] 		= pd.to_datetime(dataframe['Appt_Date'])
	dataframe['Appt_Year']		= dataframe['Appt_Date'].apply(lambda x: x.year)
	dataframe['Appt_Month']		= dataframe['Appt_Date'].apply(lambda x: x.month)
	dataframe['Appt_Day']		= dataframe['Appt_Date'].apply(lambda x: x.day)

	# Appt_Made_Time into its of year, month date
	dataframe['Appt_Made_Date'] 	= pd.to_datetime(dataframe['Appt_Made_Date'])
	dataframe['Appt_Made_Year']		= dataframe['Appt_Made_Date'].apply(lambda x: x.year)
	dataframe['Appt_Made_Month']	= dataframe['Appt_Made_Date'].apply(lambda x: x.month)
	dataframe['Appt_Made_Day']		= dataframe['Appt_Made_Date'].apply(lambda x: x.day)

	dataframe = dataframe.drop(['Cancelled','Encounter_ID','Appt_Date','Appt_Time','Appt_Made_Date',
                  'Appt_Made_Time', 'Dept_Name', 'Dept_Abbr_3', 'Dept_Abbr_4', 'Unnamed: 0'], axis=1)

	dataframe['Payor_Type_ID'].fillna(0, inplace = True)
	dataframe['Duration'].fillna((dataframe['Duration'].mean()), inplace = True)
	dataframe['Distance'].fillna((dataframe['Distance'].mean()), inplace = True)
	dataframe['Patient_Latitude'].fillna((dataframe['Patient_Latitude'].mean()), inplace = True)
	dataframe['Patient_Longitude'].fillna((dataframe['Patient_Longitude'].mean()), inplace = True)

	return dataframe


def main(group='all', no_cancel = False, one_hot = False):
	df = pd.read_csv("../data/ENCOUNTERS_RAW.csv")
	df_dept = pd.read_csv('../data/DEPT_RAW.csv')
	df = df.merge(df_dept, on = 'Dept_ID') 
	df = google_distance(df)
	df = edit(df)

	#only look at those with history
	# df = df[df['count_app'] == 1]

	df['Payor_Type_ID'].astype(int).astype('category')
	df['Dept_ID'].astype('category')
	df['Provider_ID'].astype('category')
	df['Appt_Logistics_Type_ID'].astype('category')
	df['Visit_Type_ID'].astype('category')

	#determine one hot encoding for variables:
	if one_hot == 'True':
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
	# df_count	= df.groupby('Sibley_ID').size
	# df_hist		= pd.DataFrame(values = [df_count.index,(df_count > 1)], columns = 'Sibley_ID, hist')
	 
	# df			= df.merge(df_hist, on = 'Sibley_ID')
	df['count'] = df.groupby('Sibley_ID')['Sibley_ID'].transform('count')

	if group == 'historical':
		df = df[df['count'] > 1]
	elif group == 'nonhistorical':
		df = df[df['count'] == 1]

	#get rid of cancellations
	if no_cancel == 'True':
		df = df[(df['Appt_Status_ID'] == 4) | (df['Appt_Status_ID'] == 2)]

	#df[['Payor_Type_ID']] = df[['Payor_Type_ID']].fillna(value=0)
	df = df.drop([ 'Num_Canceled_Encounters_Since',
                       'Num_No_Show_Encounters_Since',
                       'Num_Canceled_Encounters_AllTime',
                       'Num_No_Show_Encounters_AllTime',
                       'Appt_Status_ID'],
                         axis = 1)
                    
	print(df.keys())
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