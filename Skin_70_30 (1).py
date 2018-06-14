# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 00:58:28 2017


"""
import pandas as pd
import numpy as np
import csv
from sklearn import model_selection
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt 
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm 

# SKIN NON SKIN 70/30 

url = "C:\\Users\manik\Desktop\ML Assignment 1\datasets\SkinNonSkin.csv"
names = ['B', 'G', 'R', 'class']
#skin=SkinNonSkin
skin = pd.read_csv(url, sep= ';', names=names)
#Shape of our data set : Online News Popularity
print(skin.shape)
#Definition of the data (X) and the target (Y)
X = skin.iloc[:,0:3]
Y = skin.iloc[:,3]
#Spliting the data into training and testing using 70/30 hold out method
train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
#Shapes of the training datas and the testing datas
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

#Please comment the algorithms you don't want to use.

####### KNeighbors Classifier ##########
knn= KNeighborsClassifier()
#Fitting the model with our training data
print(knn.fit(X_train,Y_train))
Y_pred = knn.predict(X_test)
#### Getting the metric results 
#Mean Squared error 
print("Mean Squared Error : %.5" % sqrt(mean_squared_error(Y_test,Y_pred)))
#Mean absolute error
print("Mean Absolute Error : %.5" % mean_absolute_error(Y_test,Y_pred))


#### Logistic Regression #########

linear = linear_model.LinearRegression()
#Fitting the model with our training data
print(linear.fit(X_train,Y_train))
Y_pred = linear.predict(X_test)
#### Getting the metric results 
#Mean Squared error 
print("Mean Squared Error : %.5" % sqrt(mean_squared_error(Y_test,Y_pred)))
#Mean absolute error
print("Mean Absolute Error : %.5" % mean_absolute_error(Y_test,Y_pred))


##### Decision Tree classifier ########

tree = DecisionTreeClassifier()
#Fitting the model with our training data
print(tree.fit(X_train, Y_train))
y_pred = tree.predict(X_test)
#### Getting the metric results 
#Mean Squared error 
print("Mean Squared Error : %.5" % sqrt(mean_squared_error(Y_test,Y_pred)))
#Mean absolute error
print("Mean Absolute Error : %.5" % mean_absolute_error(Y_test,Y_pred))
