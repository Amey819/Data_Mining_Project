# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:00:16 2018

@author: araiker
"""
""" ------------------------------Import libraries  ------------------------------------"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score



### Load wine quality data into Pandas
df_red = pd.read_csv("C:/Users/imame/Desktop/Data Mining Project/winequality-red.csv")  # input the red wine dataset

print(df_red.head()) ## To get a peek of the dataset

print(df_red.info()) # to understand total count, no of null values, data type

# Let's do some plotting to know how the data columns are distributed in the dataset

#Here we see that fixed acidity does not give any specification to classify the quality.
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = df_red)

#Here we see that its quite a downing trend in the volatile acidity as we go higher the quality 
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = df_red)

#Composition of citric acid go higher as we go higher in the quality of the wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = df_red)

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = df_red)

#Composition of chloride also go down as we go higher in the quality of the wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = df_red)

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = df_red)

# Data Pre-processing 
#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
df_red['quality'] = pd.cut(df_red['quality'], bins = bins, labels = group_names) # 

""" pd.cut divides the quality 2-6.5 as bad 
     6.5 - 8 as good """
     

#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()


#Bad becomes 0 and good becomes 1 
df_red['quality'] = label_quality.fit_transform(df_red['quality']) # To encode the labels as 0 or 1
df_red['quality'].value_counts()  ## Get the count of each label

sns.countplot(df_red['quality']) # Craet a bar plot to know the values distribution of the quality

#Now seperate the dataset as response variable and feature variabes
X = df_red.drop('quality', axis = 1)  # Get all the columns except the last one
y = df_red['quality']             # get the last column as the label set

#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) 
#Applying Standard scaling to get optimized result
sc = StandardScaler() # Normalize the values so that diff between 2 column values isnt significant
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

dtree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dtree.fit(X_train, y_train)
pred_dtree = dtree.predict(X_test)
#Let's see how our model performed
print("--------------------------DecisionTreeClassifier---------------------")
print(classification_report(y_test, pred_dtree))

#Confusion matrix for the random forest classification
print(confusion_matrix(y_test, pred_dtree))
dtree_cm = confusion_matrix(y_test,pred_dtree)
print("Confusion matrix on DecisionTree: ",dtree_cm)
dtree_score = accuracy_score(y_test,pred_dtree)
print("Accuracy on DecisionTree: ",dtree_score*100)
"""
sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)

print(classification_report(y_test, pred_sgd))

print(confusion_matrix(y_test, pred_sgd))
"""

svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
print("---------------------Support Vecotr Machine ------------------------")
print(classification_report(y_test, pred_svc))
svm_cm = confusion_matrix(y_test,pred_svc)
print("Confusion matrix for support vector machine",svm_cm)
svm_score = accuracy_score(y_test,pred_svc)
print("Accuracy for support vector machine",svm_score*100)
#Finding best parameters for our SVC model
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)

grid_svc.fit(X_train, y_train)

#Best parameters for our svc model
grid_svc.best_params_

#Let's run our SVC again with the best parameters.
svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)

print(classification_report(y_test, pred_svc2))


#Now lets try to do some evaluation for random forest model using cross validation.
"""
print("---------------------Random forest cross-validation ------------------------")
dtree_eval = cross_val_score(estimator = dtree, X = X_train, y = y_train, cv = 10)
dtree_eval.mean()

#
"""





