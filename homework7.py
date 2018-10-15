# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 04:17:37 2018
IE-598 Machine Learning Assignment_7
Ensembling
@author: Haichao Bo
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

estimators = [1, 11, 25, 35, 45, 60]
for estimator in estimators:
    forest = RandomForestClassifier(n_estimators=estimator, random_state=1)
    forest.fit(X_train, y_train)
    scores = cross_val_score(estimator=forest, X=X_train, y=y_train, cv=10, n_jobs=1)
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    print('accuracy scores: %s' % forest.score(X_test, y_test))

#Display the individual feature importances of your best model using the code presented in Chapter 4 on page 136.    
feat_labels = df_wine.columns[1:]
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

print("My name is Haichao Bo")
print("My NetID is: hbo2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
