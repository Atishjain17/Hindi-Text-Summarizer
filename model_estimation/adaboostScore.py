from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd

'''
:global: none
:local: asd, x, y, param_grid, clf
:return : none
:Used to calculate the adaboost score for the dataset using its features from 'finalDataset.csv'
'''
asd = pd.read_csv("NLPData/finalDataset.csv")
x = asd.as_matrix(columns =  asd.columns[1:13])
y = asd.as_matrix(columns =  asd.columns[-1:])
y = np.squeeze(y);
param_grid = { "n_estimators" : [25, 50, 100, 150, 200, 250, 300, 350, 500], "learning_rate" : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] }
clf = GridSearchCV(AdaBoostClassifier(random_state = 0), param_grid , cv = 5, scoring = "accuracy")
clf.fit(x,y)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)