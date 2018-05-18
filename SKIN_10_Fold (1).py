# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 00:38:08 2017

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
from math import sqrt 
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation 
from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import KFold
from sklearn import svm 

# SKIN NON SKIN 10 FOLD

url = "C:\\Users\Rachid\Desktop\ML Assignment 1\datasets\SkinNonSkin.csv"
names = ['B', 'G', 'R', 'class'] # B : Bleu / G : Green / R : Red / class : 1 or 2.
#skin=SkinNonSkin
skin = pd.read_csv(url, sep= ';', names=names)
#Shape of our data set : Online News Popularity
print(skin.shape)
#Definition of the data (X) and the target (Y)
X = skin.iloc[:,0:3]
Y = skin.iloc[:,3]
print(X.shape)
print(Y.shape)
#Spliting the data into training and testing using K Fold cross Validation  method
cross_val_score

#Please comment the algorithms you don't want to use.

'''####### KNeighbors Classifier ##########
knn= KNeighborsClassifier()
#Fitting the model, metric r2

cross = cross_val_score(knn, X, Y, scoring='r2')
pred = cross_val_predict(knn,X,Y,cv=10)
print("R^2: %.3f" % r2_score(Y,pred))




'''#### Logistic Regression #########

linear = linear_model.LinearRegression()

#Fitting the model metric r2

cross = cross_val_score(linear, X, Y, scoring='r2')
pred = cross_val_predict(linear,X,Y,cv=10)
print("R^2: %.3f" % r2_score(Y,pred))



##### Decision Tree classifier ########

'''tree = DecisionTreeClassifier()

#Fitting the model, metric r2 

cross = cross_val_score(tree, X, Y, scoring='r2') 
pred = cross_val_predict(tree,X,Y,cv=10) 
print("R^2: %.3f" % r2_score(Y,pred))'''




