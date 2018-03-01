# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:50:58 2017

@author: Administrator
"""

import pandas as pd
import os
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


def load_housing_data(housing_path="E:"):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

rowdata = load_housing_data()
train_set, test_set = train_test_split(rowdata, test_size=0.2, random_state=42)
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()




# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
#imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', LabelBinarizer()),
    ])
    
    

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
    
housing_prepared = full_pipeline.fit_transform(housing)
param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]
svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
grid_search.fit(housing_prepared, housing_labels)
score = grid_search.best_score_
rmse = np.sqrt(-score)
print(rmse)

print(grid_search.best_params_)
#X = imputer.fit_transform(housing_num)
#housing_tr = pd.DataFrame(X, columns=housing_num.columns,
#                          index = list(housing.index.values))
#housing_tr = pd.DataFrame(X, columns=housing_num.columns)
#housing_cat = housing["ocean_proximity"]

#encoder = LabelBinarizer()
#housing_cat_1hot = encoder.fit_transform(housing_cat)
#print(housing_cat_1hot)
