# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 01:04:46 2017


"""
import pandas as pd
import numpy as np
import csv
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from math import sqrt 


# ONLINE NEWS POPULARITY Hold out split 70/30

url = "C:\\Users\Manik\Desktop\ML Assignment 1\datasets\ONP.csv"
#onp=Online News Popularity
onp = pd.read_csv(url, sep= ';')
#Shape of our data set : Online News Popularity
print(onp.shape)
#Definition of the data (X) and the target (Y)
X = onp.iloc[:,0:59]
Y = onp.iloc[:,59]
#Spliting the data into training and testing using 70/30 hold out method
train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
#Shapes of the training datas and the testing datas
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)





####### KNeighbors Classifier ##########

knn= KNeighborsClassifier()

#Fitting the model with our training data

print(knn.fit(X_train,Y_train))
Y_pred = knn.predict(X_test)

#### Please chose which metric you want to apply

#Mean Squared error 
#Mean Squared error 
print("Mean Squared Error : %.5" % sqrt(mean_squared_error(Y_test,Y_pred)))
#Mean absolute error
print("Mean Absolute Error : %.5" % mean_absolute_error(Y_test,Y_pred))



#### Logistic Regression #########


logistic = LogisticRegression()


#Fitting the model with our training data


print(logistic.fit(X_train,Y_train))
Y_pred = logistic.predict(X_test)
#Mean Squared error 
print("Mean Squared Error : %.5" % sqrt(mean_squared_error(Y_test,Y_pred)))
#Mean absolute error
print("Mean Absolute Error : %.5" % mean_absolute_error(Y_test,Y_pred))



##### Decision Tree classifier ########

tree = DecisionTreeClassifier()

#Fitting the model with our training data


print(tree.fit(X_train, Y_train))
Y_pred = tree.predict(X_test)
#Mean Squared error 
print("Mean Squared Error : %.5" % sqrt(mean_squared_error(Y_test,Y_pred)))
#Mean absolute error
print("Mean Absolute Error : %.5" % mean_absolute_error(Y_test,Y_pred))






