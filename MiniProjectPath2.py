import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


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

tempData = dataset_2[['Low Temp', 'High Temp', 'Precipitation']]
totalData = dataset_2['Total']

# training and testing sets of data
tempTrain, tempTest, totalTrain, totalTest = train_test_split(tempData, totalData, test_size = 0.2, random_state = 42)

linearModel = LinearRegression()
linearModel.fit(tempTrain, totalTrain)

# making predictions
totalPred = linearModel.predict(tempTest)

# calculate correlation
dataCorrelation = np.corrcoef(totalTest, totalPred)[0, 1]
print("Correlation Analysis for Temp Data ", round(dataCorrelation, 6))

# the print statement shows that the correlation of predicted and actual data is 0.763 which doesn't correspond to a strong correlation, 
# meaning that the model's predictions are not very similar to the actual values 
# so they can use the data but the results won't be extremelly accurate but pointing to the right direction

# QUESTION 3