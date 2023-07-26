# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 20:25:22 2023

@author: ATISHKUMAR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load the dataset
dataset=pd.read_csv(r'D:\Naresh_it_praksah_senapathi\jule\29th_jule\2.LOGISTIC REGRESSION CODE\Social_Network_Ads.csv')

#split the dataset into i.v and d.v
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

#split the data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.20,random_state=0)

#apply feature scaling to data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#data preproceesing done here

#apply logistic regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(penalty='l2',solver='liblinear')
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

#we apply confusion matrix to model to evalute the model
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred) 
print(cm)

#find model accuracy
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
print(ac)

#to find how much percentange of misclassification happend in this model

bias=lr.score(x_train, y_train)
bias

varience=lr.score(x_test, y_test)

#we get classification report

from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
cr

#visulizng the model

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, lr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, lr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()