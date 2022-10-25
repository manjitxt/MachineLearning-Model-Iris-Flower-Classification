# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#Loading Iris data
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'

col_name = ['sepal-lenght','sepal-width','petal-lenght','petal-width','class']

dataset = pd.read_csv(url, names = col_name)

#Summarize the Dataset
print("Summarize the Dataset :")
dataset.shape

print("displaying first 5 records of dataset:")
dataset.head()

print("Number of rows that belongs to each class:")
dataset['class'].value_counts()

# Data Visualization

sns.violinplot(y='class', x='sepal-lenght', data=dataset, inner='quartile')
plt.show()
sns.violinplot(y='class', x='sepal-width', data=dataset, inner='quartile')
plt.show()
sns.violinplot(y='class', x='petal-lenght', data=dataset, inner='quartile')
plt.show()
sns.violinplot(y='class', x='petal-width', data=dataset, inner='quartile')
plt.show()

print("Plotting multiple pairwise bivariate distributions in a dataset using pairplot: ")
sns.pairplot(dataset, hue='class', markers='+')
plt.show()

print("Plotting the heatmap to check the correlation : ")
plt.figure(figsize=(7,5))
sns.heatmap(dataset.corr(), annot=True, cmap='cubehelix_r')
plt.show()

#Model Building

print("Splitting the dataset : ")

X = dataset.drop(['class'], axis=1)
y = dataset['class']
print(f'X shape: {X.shape} | y shape: {y.shape} ')

#Train Test split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

#Model Creation

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC(gamma='auto')))
# evaluate each model in turn
results = []
model_names = []
for name, model in models:
  kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
results.append(cv_results)
model_names.append(name)
print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

model = SVC(gamma='auto')
model.fit(X_train, y_train)
prediction = model.predict(X_test)

#Printing out the classification report 

print(f'Test Accuracy: {accuracy_score(y_test, prediction)}')
print(f'Classification Report: \n {classification_report(y_test, prediction)}')
