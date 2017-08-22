from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd

'''
:global: none
:local: asd, x, y, param_grid, clf
:return : none
:Used to calculate the Random Forest score for the dataset using its features from 'finalDataset.csv'
'''
asd=pd.read_csv("NLPData/finalDataset.csv")
x=asd.as_matrix(columns= asd.columns[1:8])
y=asd.as_matrix(columns= asd.columns[-1:])
y=np.squeeze(y);
param_grid = { "n_estimators" : [100, 200, 250, 300, 350, 500, 700, 900, 1000], "max_depth" : [4, 5, 6, 7, 8, 9, 10, 11, 12] }
clf = GridSearchCV(RandomForestClassifier(random_state=0), param_grid , cv=5, scoring="accuracy")
clf.fit(x,y)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)