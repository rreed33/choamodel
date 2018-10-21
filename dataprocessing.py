import argparse
import sklearn
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
import math
import sklearn.preprocessing
import os

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
	dist_df['distance_google'] = scaler.fit_transform(dist_df['Distance'].values.reshape(-1,1))
	dist_df['duration_google'] = scaler.fit_transform(dist_df['Duration'].values.reshape(-1,1))

	#join the two dataframes on 'Encounter_ID'
	df = df.merge(dist_df, how = 'left', on='Encounter_ID')

	return df

def house_income(df):
	#this function joins the choa data, the reverse geocode data and census income data
	df_zip = 	pd.read_csv("../data/rev_geocode_all.csv")
	df_income = pd.read_csv("../data/income_by_zip.csv")

	df_income['Annual payroll ($1,000)'].astype(float)
	df_income['Paid employees for pay period including March 12 (number)'].astype(int)
	df_income['mod_income'] = 1000 * df_income['Annual payroll ($1,000)']/ df_income['Paid employees for pay period including March 12 (number)']
	df_income.drop([i for i in df_income.keys() if i not in ['ID2', 'mod_income']])
	print(df_income['mod_income'])

	df = df.merge(df_zip, on = "Encounter_ID", how = "left")
	df = df.merge(df_income, left_on = "Address", right_on = "Id2", how = "left")

	return df 

def edit(dataframe):
	#use this function to organize all the edits to the raw data before processing


	#get the bird's eye distance to from home to office
	dataframe['distance_bird'] = np.vectorize(distance)(dataframe['Patient_Latitude'], -1*dataframe['Patient_Longitude'],
						 				dataframe['Dept_Location_Latitude'], -1*dataframe['Dept_Location_Longitude'] )
	dataframe['distance_google'].fillna(dataframe['distance_bird'])

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
	dataframe['diff_pay_count']	= grouped_sibley['Payor_Type_ID'].apply( lambda x: ((x - x.shift()) != 0).cumsum() - 1 )

	# Appt_Made_Time into its of year, month date
	dataframe['Appt_Made_Date'] 	= pd.to_datetime(dataframe['Appt_Made_Date'])
	dataframe['Appt_Made_Year']		= dataframe['Appt_Made_Date'].apply(lambda x: x.year)
	dataframe['Appt_Made_Month']	= dataframe['Appt_Made_Date'].apply(lambda x: x.month)
	dataframe['Appt_Made_Day']		= dataframe['Appt_Made_Date'].apply(lambda x: x.day)

	dataframe['Appt_Made_Time'] 	= pd.to_datetime(dataframe['Appt_Made_Time'])
	dataframe['Appt_Made_Hour']		= dataframe['Appt_Made_Time'].apply(lambda x: x.hour)
	dataframe['Appt_Made_Min']		= dataframe['Appt_Made_Time'].apply(lambda x: x.minute)

	# Appt_Made_Time into its of year, month date
	dataframe['Appt_Date'] 		= pd.to_datetime(dataframe['Appt_Date'])
	dataframe['Appt_Year']		= dataframe['Appt_Date'].apply(lambda x: x.year)
	dataframe['Appt_Month']		= dataframe['Appt_Date'].apply(lambda x: x.month)
	dataframe['Appt_Day']		= dataframe['Appt_Date'].apply(lambda x: x.day)

	dataframe['Appt_Time']			= pd.to_datetime(dataframe['Appt_Time'])
	dataframe['Appt_Time_Hour']		= dataframe['Appt_Time'].apply(lambda x: x.hour)
	dataframe['Appt_Time_Min']		= dataframe['Appt_Time'].apply(lambda x: x.minute)

	dataframe = dataframe.drop(['Cancelled','Encounter_ID','Appt_Date','Appt_Time','Appt_Made_Date',
                  'Appt_Made_Time', 'Dept_Name', 'Dept_Abbr_3', 'Dept_Abbr_4', 'Unnamed: 0'], axis=1)

	dataframe['Payor_Type_ID'].fillna(0, inplace = True)
	dataframe['Duration'].fillna((dataframe['Duration'].mean()), inplace = True)
	dataframe['distance_google'].fillna((dataframe['distance_google'].mean()), inplace = True)
	dataframe['distance_bird'].fillna((dataframe['distance_bird'].mean()), inplace = True)
	dataframe['Patient_Latitude'].fillna((dataframe['Patient_Latitude'].mean()), inplace = True)
	dataframe['Patient_Longitude'].fillna((dataframe['Patient_Longitude'].mean()), inplace = True)


	return dataframe


def main(group='all', no_cancel = False, one_hot = False, original = False, generate_data = 'False', office = 'all'):
	# READ FROM INTERMEDIATE FILES OF SIMILAR DATA FORMULATIONS
	intermediate_data_name = '../data/choa_group_{}_no_cancel_{}_one_hot_{}_original_{}_office_{}intermediate.csv'.format(
				group, no_cancel, one_hot, original, office)

	if generate_data == 'False' and os.path.exists(intermediate_data_name):
		print('\nREADING FROM FILE ', intermediate_data_name, '\n--------------------\n\n')
		df = pd.read_csv(intermediate_data_name)
		return df
	elif generate_data == 'False' and not os.path.exists(intermediate_data_name):
		print('\nTHIS FORMULATION HAS NOT BEEN RECORDED\nCONTINUING TO GENERATE DATA FROM RAW DATA\n--------------------\n\n')
	elif generate_data == 'True' and os.path.exists(intermediate_data_name):
		print('\nTHIS FORMULATION COULD HAVE BEEN DONE FASTER IF YOU HAD SET generate_date TO False\n--------------------\n\n')

	#focus on the chosen location
	
	office_code = {'augusta': 4, 'canton': 5,'columbus':6,'cumming':7,'dalton':8,'emory':9,'gainesville':11,'hamilton mill':12,
						'johns creek':13,'macon':14,'marietta':15,'newnan':16,'scottish rite':17,'snellville':18,'stockbridge':19,
						'thomasville':20,'tifton':21,'valdosta':22,'villa rica':23,'egleston':24,'lawrenceville':26,'rockdale':27}

	df = pd.read_csv("../data/ENCOUNTERS_RAW.csv")
	df_dept = pd.read_csv('../data/DEPT_RAW.csv')

	if office in office_code.keys():
		df = df[df['Dept_ID']==office_code[office]]
	elif office not in office_code.keys() and office != 'all':
		print('ERROR: a specific office was not identified. will continue model with full data set')
		
	df = df.merge(df_dept, on = 'Dept_ID') 
	df = google_distance(df)
	df = edit(df)
	df = house_income(df)



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
	df['count'] = df.groupby('Sibley_ID')['Sibley_ID'].transform('count')
	if group == 'historical':
		df = df[ df['count'] > 1 ]
	elif group == 'nonhistorical':
		df = df[ df['count'] == 1 ]

	#get rid of cancellations
	if no_cancel == 'True':
		df = df[(df['Appt_Status_ID'] == 4) | (df['Appt_Status_ID'] == 2)]

	#df[['Payor_Type_ID']] = df[['Payor_Type_ID']].fillna(value=0)
	df = df.drop([ 'Num_Canceled_Encounters_Since',
                       'Num_No_Show_Encounters_Since',
                       'Num_Canceled_Encounters_AllTime',
                       'Num_No_Show_Encounters_AllTime',
                       'Appt_Status_ID', 'Patient_Longitude', 'Patient_Latitude',
                       'Dept_Location_Longitude', 'Dept_Location_Latitude'],
                         axis = 1)

	if original == 'True':
		print('dropped')
		df = df.drop(['count_app', 'count_cancel', 'count_miss', 'distance_bird',
					'duration_google', 'distance_google', 'diff_pay_count'], axis = 1)

	print('CHECK FEATURES:')
	print(df.keys())
	print()
	df.to_csv('../data/choa_group_{}_no_cancel_{}_one_hot_{}_original_{}_office_{}intermediate.csv'.format(
				group, no_cancel, one_hot, original, office))
	return df

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-original', type =str, default = 'False',
			help = 'set equal to True to reduce data to original form')
	parser.add_argument('-office', type = str, default = 'macon')
	args = parser.parse_args()

	main(args.original, args.office)


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