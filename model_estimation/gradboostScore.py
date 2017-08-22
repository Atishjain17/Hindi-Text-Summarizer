from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd

'''
:global: none
:local: asd, x, y, param_grid, clf
:return : none
:Used to calculate the gradboost score for the dataset using its features from 'finalDataset.csv'
'''
if __name__ == "__main__":
    asd=pd.read_csv("NLPData/finalDataset.csv")
    x=asd.as_matrix(columns= asd.columns[1:13])
    y=asd.as_matrix(columns= asd.columns[-1:])
    y=np.squeeze(y);
    param_grid = { "loss" : ['deviance', 'exponential'], "n_estimators" : [25, 50, 100, 200, 250, 300, 400, 500],
               "learning_rate" : [0.1, 0.05, 0.01],"max_depth" : [3, 4, 5, 6, 7, 8, 9, 10, 11, 12] }
    clf = GridSearchCV(GradientBoostingClassifier(random_state=0), param_grid , cv=2,
                       scoring="accuracy", verbose=1, n_jobs=3)
    clf.fit(x,y)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)