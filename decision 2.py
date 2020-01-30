# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 06:42:23 2020

@author: soube
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import  metrics
from sklearn.metrics import accuracy_score
import graphviz
data=pd.read_csv("C:\\Users\\soube\\Downloads\\train_and_test2.csv")
data=data [["Passengerid","Age","Fare","Sex","sibsp","Parch","Pclass","Embarked","2urvived"]]
data['Passengerid']=data.Passengerid.fillna(data.Passengerid.mean())
data['Age']=data.Age.fillna(data.Age.mean())
data['Fare']=data.Fare.fillna(data.Fare.mean())
data['Sex']=data.Sex.fillna(data.Sex.mean())
data['sibsp']=data.sibsp.fillna(data.sibsp.mean())
data['Parch']=data.Parch.fillna(data.Parch.mean())
data['Pclass']=data.Pclass.fillna(data.Pclass.mean())
data['Embarked']=data.Embarked.fillna(data.Embarked.mean())
x=data[["Passengerid","Age","Fare","Sex","sibsp","Parch","Pclass","Embarked"]]
y=data['2urvived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)
print(x_train)
print(y_train)
clf=RandomForestClassifier(n_estimators=500)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
A=pd.DataFrame()
A=pd.concat([x_train,y_train])
B=pd.DataFrame()
B=pd.concat([x_test,y_test])
train_data=A
test_data=B
train_labels=data['2urvived']
classifier=tree.DecisionTreeClassifier()
classifier.fit(train_data,train_labels)
predicted=classifier.predict(test_data)
print('Score:{}'.format(classifier.score(train_data,train_labels)))
dot_data=tree.export_graphviz(classifier,out_file=None)
graph=graphviz.Source(dot_data)
graph.render("data")
graph
dtree=tree.DecisionTreeClassifier(criterion="gini",splitter='random',max_leaf_nodes=10,min_samples_leaf=5,max_depth=5)
