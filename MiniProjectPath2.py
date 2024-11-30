import pandas
import numpy as np

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Total']  = pandas.to_numeric(dataset_2['Total'].replace(',','', regex=True))
#  print(dataset_2.to_string()) #This line will print out your data

# data preprocesssing
dataset_2['Date'] = pandas.to_datetime(dataset_2['Date'] + '-2016', format='%d-%b-%Y')  # convert the date
dataset_2['Day of Week'] = dataset_2['Date'].dt.day_name()  # get the day names
bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge']

# QUESTION 1
# getting the correlation between each bridge's traffic and total traffic
correlations = dataset_2[bridges + ['Total']].corr()['Total'].drop('Total')
print("Correlation Analysis Results")
print(correlations)

# from the print statement above, we see that Manhattan, Queensboro, and Williamsburg have the highest correlations
# as a result, we should put sensors on all the bridges except for the Brooklyn Bridge

# QUESTION 2

# QUESTION 3