# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 00:51:05 2017

@author: Rachid
"""
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict
from numpy import array
from numpy import argmax
from sklearn import metrics 
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import datasets

url = "C:\\Users\Rachid\Desktop\skin.csv"
names = ['B', 'G', 'R', 'class']
skin = pd.read_csv(url, sep= ';', names=names)
print(skin.shape)


X = skin.ix[:,0:3]
Y = skin.ix[:,3]
print(X.shape)
print(Y.shape)

      
models = []
models.append(('LR', LinearRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))


results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=0)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

