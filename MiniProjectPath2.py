import pandas
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
def three_bridges(bridge1, bridge2, bridge3):
    X = dataset_2[[f"{bridge1}", f"{bridge2}", f"{bridge3}"]].values
    y = dataset_2[["Total"]].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    PredictScore = model.score(X_test, y_test)

    return model, PredictScore

def normalize_train(X_train):
    trn_mean = np.mean(X_train, axis=0)
    trn_std = np.std(X_train, axis=0)
    X_norm = (X_train - trn_mean) / trn_std
    return X_norm, trn_mean, trn_std

def normalize_test(X_test, trn_mean, trn_std):
    X_norm = (X_test - trn_mean) / trn_std
    return X_norm

# ridge
def train_model(X_train, y_train, alpha):
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, y_train)
    return model

# analysis
def analyze_weekly_patterns_and_predict(dataset):
    weekly_avg = dataset.groupby('Day of Week')[bridges + ['Total']].mean()
    
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_avg = weekly_avg.reindex(ordered_days)
    print("Weekly Averages:")
    print(weekly_avg) 
    # visualize !!!!!!
    weekly_avg.plot(kind='bar', figsize=(12, 6))
    plt.title('Average Traffic by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Average Traffic Count')
    plt.legend(title='Bridge')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', linewidth=0.7)
    plt.show()
    
    # preparing data
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    dataset['Day Code'] = dataset['Day of Week'].map(day_map)
    
    # bridge traffic -> features
    # day code -> target
    X = dataset[bridges].to_numpy()
    y = dataset['Day Code'].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # normalizing training data!
    X_train_norm, trn_mean, trn_std = normalize_train(X_train)
    
    # normalize data yay!
    X_test_norm = normalize_test(X_test, trn_mean, trn_std)
    
    # train and evaluate
    alpha = 1.0  # the regularization parameter 
    model = train_model(X_train_norm, y_train, alpha)
    
    y_pred = model.predict(X_test_norm)
    y_pred_rounded = np.round(y_pred).astype(int)  

    mse = mean_squared_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred_rounded)
    print(f"Model MSE: {mse:.2f}")
    print(f"Model Accuracy: {accuracy:.2%}")
    
    return model, trn_mean, trn_std


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

print("QUESTION 1")
# getting the correlation between each bridge's traffic and total traffic

# bridge traffic -> feature
# total traffic -> target
scores = [0, 0, 0, 0]
mwq, scores[0] = three_bridges("Manhattan Bridge", "Williamsburg Bridge", "Queensboro Bridge")
bmq, scores[1] = three_bridges("Brooklyn Bridge", "Manhattan Bridge", "Queensboro Bridge")
bwq, scores[2] = three_bridges("Brooklyn Bridge", "Williamsburg Bridge", "Queensboro Bridge")
bmw, scores[3] = three_bridges("Brooklyn Bridge", "Manhattan Bridge", "Williamsburg Bridge")

print("Manhattan, Williamsburg, Queensboro: "+ str(scores[0]))
print("Brooklyn, Manhattan, Queensboro: "    + str(scores[1]))
print("Brooklyn, Williamsburg, Queensboro: " + str(scores[2]))
print("Brooklyn, Manhattan, Williamsburg: "  + str(scores[3]))

# from the print statement above, we see that Brooklyn, Manhattan, and Williamsburg has the highest score.
# as a result, we should put sensors on all the bridges except for the Queensboro Bridge

print("\nQUESTION 2")

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

print("\nQUESTION 3")
model, trn_mean, trn_std = analyze_weekly_patterns_and_predict(dataset_2)
# the print statement shows that the model accuracy for predicting the day through the bicycle traffic is 16.28%, indicating that
# it is not ideal to use bicycle traffic to predict the day of the week, but one could potentially predict the