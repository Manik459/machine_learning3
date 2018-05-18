# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 02:45:47 2017

@author: Rachid
"""

import pandas as pd
import numpy as np
import csv
from sklearn import model_selection
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import linear_model 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation 
from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import KFold
from sklearn import svm 
from math import sqrt

# ONLINE NEWS POPULARITY 10 FOLD

url = "C:\\Users\Rachid\Desktop\ML Assignment 1\datasets\ONP.csv"

#onp=ONLINE NEWS POPULARITY
onp = pd.read_csv(url, sep= ';')
#Shape of our data set : Online News Popularity
print(onp.shape)
#Definition of the data (X) and the target (Y)
X = onp.iloc[:,0:59]
Y = onp.iloc[:,59]
#Spliting the data into training and testing using K Fold cross Validation  method
cross_val_score

#Please comment the algorithms you don't want to use.

####### KNeighbors Classifier ##########
knn= KNeighborsClassifier()
#Fitting the model with our training data

cross = cross_val_score(knn, X, Y)
pred = cross_val_predict(knn,X,Y,cv=10)
#Mean Squared error 
print("Mean Squared Error : %.5" % sqrt(mean_squared_error(Y,pred)))
#Mean absolute error
print("Mean Absolute Error : %.5" % mean_absolute_error(Y,pred))



##'''

#### Logistic Regression #########

linear = LinearRegression()
#Fitting the model with our training data

cross = cross_val_score(linear, X, Y)
pred = cross_val_predict(linear,X,Y,cv=10)
#Mean Squared error 
print("Mean Squared Error : %.5" % sqrt(mean_squared_error(Y,pred)))
#Mean absolute error
print("Mean Absolute Error : %.5" % mean_absolute_error(Y,pred))



##### Decision Tree classifier ########

tree = DecisionTreeClassifier()

#Fitting the model, metric r2 please comment these 3 line to calculate accuracy

cross = cross_val_score(tree, X, Y,) 
pred = cross_val_predict(tree,X,Y,cv=10) 

#Mean Squared error 
print("Mean Squared Error : %.5" % sqrt(mean_squared_error(Y,pred)))
#Mean absolute error
print("Mean Absolute Error : %.5" % mean_absolute_error(Y,pred))
