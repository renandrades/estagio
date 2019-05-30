
from __future__ import print_function

from joblib import dump
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from matplotlib.mlab import griddata

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def printAccuracy(clf, Xtr, Xte):
	y_hat_test = clf.predict(Xte)
	y_hat_train = clf.predict(Xtr)
	acc_train =accuracy_score(y_train, y_hat_train)
	acc_test = accuracy_score(y_test,y_hat_test)

	print('%.2f %.2f' % (acc_train, acc_test))
	return acc_train, acc_test


data1 = pd.read_csv('data_arrhythmia.csv', sep = ';', na_values='?')
data = data1.drop(data1[(data1['height'] > 250)].index) 

for col in data.columns:
	num_null = data[col].isnull().sum()
	if num_null:
		print('Column %s has %d (%.1f%%) null values' % (col, num_null, 100.0*num_null/data.shape[0]))

for col in data.columns:
	num_unique = np.unique(data[col])
	if len(num_unique) == 1:
		print('Column %s has a single value and will be discarded' % (col))
		data.drop(columns=[col], inplace = True)

data.fillna(data.mean(), inplace = True)

data.loc[data.diagnosis == 1, 'diagnosis'] = 0
data.loc[data.diagnosis  > 1, 'diagnosis'] = 1

X = data.loc[:, data.columns != 'diagnosis']
y = data['diagnosis']


from sklearn.ensemble import RandomForestClassifier as RF 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

best_score = float('-inf')


print("Testing Random Forests")
							
clf = RF().fit(X_train, y_train)
score_tr, score_te = printAccuracy(clf, X_train, X_test)
if score_te > best_score:
	best_score = score_te
	best_rf = clone(clf)
						    
   
    
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.svm import SVC

best_score = float('-inf')



print("\nTesting SVM")

clf = SVC().fit(X_train_scaled, y_train)

score_tr, score_te = printAccuracy(clf, X_train_scaled, X_test_scaled)
if score_te > best_score:
	best_score = score_te
	best_svm = clone(clf)
                    
			        


from sklearn.ensemble import GradientBoostingClassifier as GBC

best_score = float('-inf')


print("\nTesting GBC")

clf = GBC().fit(X_train_scaled, y_train)
score_tr, score_te = printAccuracy(clf, X_train_scaled, X_test_scaled)
if score_te > best_score:
	best_score = score_te
	best_gbc = clone(clf)




