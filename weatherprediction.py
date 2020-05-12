#A machine learnIng model to predict weather it would rain tomorrow based on the data from an Australian weather datasets
import pandas as pd
import numpy as np

df = pd.read_csv("C://Users//EMMANUEL//Downloads//weather-dataset-rattle-package.zip")
print(df.shape)
print(df[0:6])
#This enables a descriptive exploration of our data
df.describe()
#Data Preprocessing 
#Checking for null values

print(df.count().sort_values())
#Removing all null values
df = df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am', 'Location', 'RISK_MM', 'Date'],axis = 1)
print(df.shape)
#Lets get rid of any null values
df = df.dropna(how='any')
print(df.shape)
#check the dataset for any ouliers and remove them. 
#An outlier is a datapoint different from the other obsedrvations.It occurs due to miscalculation while collecting the data.

from scipy import stats
z = np.abs(stats.zscore(df._get_numeric_data()))
print(z)
df = df[(z<3).all(axis=1)]
print(df.shape)
#Dealing with Categorical values now
#Simply change Yes/No to 1/0 for RainToday and RainTomorrow

df['RainToday'].replace({'No': 0, 'Yes':1}, inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes':1}, inplace = True)
#Changing unique values into integer values using pd.getDummies()

categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
for col in categorical_columns:
    print(np.unique(df[col]))
#Transform rhe categorical column
df = pd.get_dummies(df, columns = categorical_columns)
print(df. iloc[4:9])
#Normalize or Standardize our data using MinMaxScaler

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), index = df.index, columns = df.columns)
print(df.iloc[4:10])
#Exploratory data Analysis: We are going to analyses and identify the Significant variables 
#that would help us predict the outcome using SelectKBest

from sklearn.feature_selection import SelectKBest, chi2
x = df.loc[:, df.columns!= 'RainTomorrow']
y = df[['RainTomorrow']]
selector = SelectKBest(chi2, k=3)
selector.fit(x, y)
x_new = selector.transform(x)
print(x.columns[selector.get_support(indices = True)])
#Data modelling
#Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

#Calculating the accuracy and time taken by the classifier 
t0 = time.time()

#Data splicing
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.25)
clf_logreg = LogisticRegression(random_state=0)

#Building the model using the training dataset,this will fit your training dataset
clf_logreg.fit(x_train,y_train)

#Evaluating the model using testing dataset
y_pred = clf_logreg.predict(x_test)
score = accuracy_score(y_test,y_pred)

#printing the accuracy and the time taken by the clasifier 
print('Accuracy using Logistic Regression:',score)
print('The time taken using Logistic Regression:', time.time()-t0)
#Random Forest classifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Calculating the accuracy and the time taken by the classifier
t0 = time.time()

#Data splicing 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
clf_rf = RandomForestClassifier(n_estimators=100,max_depth=4,random_state=0)

#Building the model using training dataset \
clf_rf.fit(x_train,y_train)

#Evaluating the model testing dataset
y_pred = clf_rf.predict(x_test)
score = accuracy_score(y_test,y_pred)

#Print the accuracy and the time taken by the classifier
print('Accuracy using the Random forest classifier:',score)
print('The time taken using the Random forest classifier:', time.time()-t0)
#Support Vector Machine 

from sklearn import svm
from sklearn.model_selection import train_test_split
import time

#Calculating the accuracy and the time taken by the classifier
t0 = time.time()

#Data splicing 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
clf_svc = svm.SVC(kernel = 'linear')

#Building the model using training dataset \
clf_svc.fit(x_train,y_train)

#Evaluating the model testing dataset
y_pred = clf_svc.predict(x_test)
score = accuracy_score(y_test,y_pred)

#Print the accuracy and the time taken by the classifier
print('Accuracy using the Random forest classifier:',score)
print('The time taken using the Random forest classifier:', time.time()-t0)
#Decision Tree Classifier 
from sklearn import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#Calculating the accuracy and the time taken by the classifier
t0 = time.time()

#Data splicing 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
clf_dt = DecisionTreeClassifier(random_state=0)

#Building the model using training dataset \
clf_dt.fit(x_train,y_train)

#Evaluating the model testing dataset
y_pred = clf_dt.predict(x_test)
score = accuracy_score(y_test,y_pred)

#Print the accuracy and the time taken by the classifier
print('Accuracy using the Decision Tree classifier:',score)
print('The time taken using the Decision Tree classifier:', time.time()-t0)
