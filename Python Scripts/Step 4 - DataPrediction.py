#*******************************************
# Author: Kevin Fitzgerald
# Project Purpose: Predicting USA golf course grass types
# Date: November 2018
#*******************************************

import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sn

#**************************************************************

# OPEN DATASET AND EXAMINE ITS STRUCTURE/STATISTICS
courses = pd.read_csv("/Users/kf/Desktop/finaltable2.csv")
print("\nCount of Null Values per Column:\n",courses.isnull().sum())
print("\nData Column Types:\n",courses.info())
print("\nGreen Label Frequencies\n",courses['Green Type'].value_counts())
print("\nFairway Label Frequencies\n",courses['Fairway Type'].value_counts())

#**************************************************************

# TRANSFORM SLOPE, RATING, AND AVAILABILITY FIELDS
Slope=[]
Rating=[]
numbers=["1","2","3","4","5","6","7","8","9","0"]
for i in courses['Slope']:
    i=str(i)
    #If there are numbers in the value then split it and take the Men's value
    if any(num in i for num in numbers):
        value = i[:i.find('Ladies')]
        value = value.replace("Men's"," ")
        value = value.replace(" ","")
        value = int(value)
        Slope.append(value)
    else:
        i=float('nan')
        Slope.append(i)

Slope = pd.DataFrame(Slope,columns=['Slope'])
Slope = Slope.fillna(Slope.mean())
Slope = Slope.astype('int64')

for i in courses["Rating"]:
    i=str(i)
    #If there are numbers in the value then split it and take the Men's value
    if any(num in i for num in numbers):
        value = i[:i.find('Ladies')]
        value = value.replace("Men's"," ")
        value = value.replace(" ","")
        value = float(value)
        Rating.append(value)
    else:
        i=float('nan')
        Rating.append(i)

Rating = pd.DataFrame(Rating,columns=['Rating'])
Rating = Rating.fillna(Rating.mean())
Rating = Rating.round(2)
Rating = Rating.astype('float')

courses = courses.drop(columns=['Slope', 'Rating'])
courses = pd.concat([courses, Slope, Rating],axis=1)

courses = courses.dropna(how='any')

# Define Green Type column and Fairway Type column as TARGET variables
# and all other columns as our PREDICTIVE variables
x = courses.iloc[:,courses.columns != "Green Type"]
x = x.iloc[:,x.columns != "Fairway Type"]
y1 = courses.iloc[:,courses.columns == "Green Type"]
y2 = courses.iloc[:,courses.columns == "Fairway Type"]
y = pd.concat([y1, y2],axis=1)

# Drop Name, Zip Code, and Address columns because they are irrelevant details
# Also drop the Country column because all courses are in the USA
x = x.drop(columns=['Name', 'Country', 'Address', 'Zip Code', 'Designer','City'])

# Transform Season Availibility field
# Turn values from May to Nov into 7 Months
Avail=[]
for i in x['Season Availability']:
    if "year" in i or "Year" in i:
        i = '12'
        Avail.append(i)
    else:
        i = re.sub('\(.*?\)', '', i)
        i = re.sub('[0-9]','',i)
        if i == 'Jan  to Dec ' or i == 'Jan  to Jan ' or i == 'Jan  to Jan  ':
            i = '12'
            Avail.append(i)
        if 'Year' not in i:
            i = i.replace('Jan','1')
            i = i.replace('Feb','2')
            i = i.replace('Mar','3')
            i = i.replace('Apr','4')
            i = i.replace('May','5')
            i = i.replace('Jun','6')
            i = i.replace('Jul','7')
            i = i.replace('Aug','8')
            i = i.replace('Sep','9')
            i = i.replace('Oct','10')
            i = i.replace('Nov','11')
            i = i.replace('Dec','12')
            startMon = i[:i.find('to')]
            startMon = startMon.replace('to','')
            startMon = startMon.replace(' ','')
            startMon = int(startMon)
            endMon = i[i.find('to'):]
            endMon = endMon.replace('to','')
            endMon = endMon.replace(' ','')
            endMon = int(endMon)
            seasonLength = (endMon - startMon)+1
            if seasonLength == 0:
                seasonLength = '12'
                Avail.append(seasonLength)
            elif seasonLength == -1:
                seasonLength = 11
                Avail.append('%i' % seasonLength)
            elif seasonLength == -2:
                seasonLength = 10
                Avail.append('%i' % seasonLength)
            elif seasonLength == -4:
                seasonLength = 8
                Avail.append('%i' % seasonLength)
            elif seasonLength == -5:
                seasonLength = 7
                Avail.append('%i' % seasonLength)
            elif seasonLength == -6:
                seasonLength = 6
                Avail.append('%i' % seasonLength)
            else:
                Avail.append('%i' % seasonLength)
Availnum=[]
for i in Avail:
    i = int(i)
    Availnum.append(i)
del Availnum[-1]
del Availnum[-1]
del Availnum[-1]
Availnum = pd.DataFrame(Availnum,columns=['Availability'])
x = x.drop(columns=['Season Availability'])
x = x.reset_index(drop=True)
x = pd.concat([x, Availnum],axis=1)

#******************************************************************

# CONVERT CATEGORICAL VARIABLES AND FORMAT NUMERICAL VARIABLES
# Split the dataframe by type
nums = x.select_dtypes(exclude=object)
objs = x.select_dtypes(include=object)

# Format numerical columns
nums = nums.astype({"Year Established": int, "Slope": int, "Rating": float, "Availability": int,
                    "annualhigh": float,"annuallow": float, "averagetemp":float,"averageannualrain": float})

# Dummy encode categorical variables and put dataframe back together
objs = pd.get_dummies(objs)
x = pd.concat([objs,nums],axis=1)

#******************************************************************

# IDENTIFY TOP 50 MOST IMPORTANT FEATURES FOR PREDICTIVE ANALYSIS
forest = RandomForestClassifier(random_state=100)
forestfit = forest.fit(x,y)
important_features = pd.Series(data=forestfit.feature_importances_,index=x.columns)
important_features.sort_values(ascending=False,inplace=True)

# Select the Top 50 features for analysis
top50 = important_features[:50]

# Drop any features that aren't in the Top 50
top50columns = list(top50.index)
columns = list(x)
for i in columns:
    if i not in top50columns:
        x = x.drop([i],axis=1)
    else:
        pass

#Create Dataframe for plot
top50df = pd.DataFrame(
    {'Feature': top50.values,
     'Importance Score': top50.index,
    }
)

# Plot Top 50 Features and their importance scores as a horizontal bar graph
plt.figure(figsize = (10,7))
seabar = sn.barplot(x='Feature', y='Importance Score', data=top50df)
seabar.set_xlabel('Importance Score')
seabar.set_ylabel('Feature Name')
plt.title('Top 50 Feature Importance Scores')
plt.show()
    
#******************************************************************

# Split the data into train and test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=1)

# Establish a Random Forest Classifier
forest = RandomForestClassifier(random_state=100)

# Establish parameter grid used to find the best combination
param_grid = { 
    'n_estimators': [100,250,500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy'],
    'class_weight': ['balanced','balanced_subsample']
}
# Establish a GridSearchCV variable with the classifier and paramter grid
Grid = GridSearchCV(estimator=forest, param_grid=param_grid, cv= 5)
Grid2 = GridSearchCV(estimator=forest, param_grid=param_grid, cv= 5)

# Fit the GridSearchCV with the training data
Gridfit = Grid.fit(xtrain,ytrain['Green Type'])
Gridfit2 = Grid2.fit(xtrain,ytrain['Fairway Type'])

print("Best Parameters for Predicting Green Turf", Grid.best_params_)
print("Best Parameters for Predicting Fairway Turf", Grid2.best_params_)

# Get the best model, use it in the Multi-Output Classifier, and make predictions
bestforest = Grid.best_estimator_
multi_target = MultiOutputClassifier(bestforest, n_jobs=1)
preds = multi_target.fit(xtrain, ytrain).predict(xtest)

preds = pd.DataFrame(preds,columns=['Green Type','Fairway Type'])
print(preds)

# Encode target data and predictions in numerics so they can be plugged into scoring metrics
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
labels = pd.concat([y['Green Type'],y['Fairway Type']],axis=0)
lefit = le.fit(labels)
ymatrix = ytest.copy()
ymatrixp = preds.copy()
ytest['Green Type'] = lefit.transform(ytest['Green Type'])
ytest['Fairway Type'] = lefit.transform(ytest['Fairway Type'])
preds['Green Type'] = lefit.transform(preds['Green Type'])
preds['Fairway Type'] = lefit.transform(preds['Fairway Type'])

# Assess model accuracy and performance
print("Green Turf Accuracy Score: ",accuracy_score(ytest['Green Type'],preds['Green Type']))
print("Fairway Turf Accuracy Score: ",accuracy_score(ytest['Fairway Type'],preds['Fairway Type']))
print("Aggregate Accuracy Score: ",(accuracy_score(ytest['Fairway Type'],preds['Fairway Type'])+accuracy_score(ytest['Green Type'],preds['Green Type']))/2)
print("Explained Variance Score: ", explained_variance_score(ytest, preds))
print("Mean Absolute Error Score: ", mean_absolute_error(ytest, preds, multioutput='uniform_average'))
print("Classification Report - Green Turf\n", classification_report(ytest['Green Type'],preds['Green Type'],target_names=ymatrix['Green Type'].unique()))
print("Classification Report - Fairway Turf\n", classification_report(ytest['Fairway Type'],preds['Fairway Type'],target_names=ymatrix['Fairway Type'].unique()))

#This code uses the true and pred prefixes so we know which is which
#cm = pd.DataFrame(confusion_matrix(y_true=ymatrix['Green Type'], y_pred=ymatrixp['Green Type'], labels=labels.unique()), 
#                   index=['true:{:}'.format(x) for x in labels.unique()], 
#                   columns=['pred:{:}'.format(x) for x in labels.unique()])

cm = pd.DataFrame(confusion_matrix(y_true=ymatrix['Green Type'], y_pred=ymatrixp['Green Type'], labels=ymatrix['Green Type'].unique()), 
                   index=[format(x) for x in ymatrix['Green Type'].unique()], 
                   columns=[format(x) for x in ymatrix['Green Type'].unique()])

plt.figure(figsize = (10,7))
seaheat = sn.heatmap(cm, annot=True, linewidths=.5, fmt="d", cmap="Oranges")
#seaheat = sn.heatmap(cm, annot=True, linewidths=.5, fmt="d", mask=(cm==0), cmap="Oranges")
#seaheat.set_facecolor('#f0f0f0')
plt.xlabel('Predicted label')
plt.ylabel('True Label')
plt.title('Green Turf Accuracy/Confusion Matrix')
plt.show()

cm1 = pd.DataFrame(confusion_matrix(y_true=ymatrix['Fairway Type'], y_pred=ymatrixp['Fairway Type'], labels=ymatrix['Fairway Type'].unique()), 
                   index=[format(x) for x in ymatrix['Fairway Type'].unique()], 
                   columns=[format(x) for x in ymatrix['Fairway Type'].unique()])

plt.figure(figsize = (10,7))
seaheat = sn.heatmap(cm1, annot=True, linewidths=.5, fmt="d", cmap="Oranges")
plt.xlabel('Predicted label')
plt.ylabel('True Label')
plt.title('Fairway Turf Accuracy/Confusion Matrix')
plt.show()
