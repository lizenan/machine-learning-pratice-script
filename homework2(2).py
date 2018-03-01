# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 00:00:03 2017

@author: Administrator
"""

import sklearn as sl
from sklearn import datasets as ds
from sklearn import cross_validation as cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from scipy.spatial import distance
iris = ds.load_iris()

def euc(a, b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.Y_train[best_index]
x = iris.data
y = iris.target

X_train , X_test , Y_train , Y_test = cv.train_test_split(x , y , test_size=.5)

my_classifier = ScrappyKNN()
my_classifier.fit(X_train , Y_train)
predictions = my_classifier.predict(X_test)

print (metrics.accuracy_score(Y_test , predictions))
 