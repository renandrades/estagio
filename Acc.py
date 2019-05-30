
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
	acc_train =recall_score(y_train, y_hat_train)
	acc_test = recall_score(y_test,y_hat_test)

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
from sklearn.metrics import accuracy_score, recall_score
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

best_score = float('-inf')


print("Testing Random Forests")
							
clf_t = RF(bootstrap=True,max_depth=20,max_features=100,min_samples_leaf=2,min_samples_split=20,n_estimators=50).fit(X_train, y_train)
clf = RF().fit(X_train, y_train)

print('Default')
score_tr, score_te = printAccuracy(clf, X_train, X_test)
print('Tuned')
score_tr, score_te = printAccuracy(clf_t, X_train, X_test)
				    
   
    
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
clf_t = SVC(C=128,gamma=0.0001 ,shrinking=1,tol=0.001).fit(X_train_scaled, y_train)

print('Default')
score_tr, score_te = printAccuracy(clf, X_train_scaled, X_test_scaled)
print('Tuned')
score_tr, score_te = printAccuracy(clf_t, X_train_scaled, X_test_scaled)
                    
			        


from sklearn.ensemble import GradientBoostingClassifier as GBC


print("\nTesting GBC")

clf_t = GBC(learning_rate=0.01,loss='exponential',max_depth=10,max_features='auto',min_samples_leaf=1,min_samples_split=50,n_estimators=600,subsample=0.5).fit(X_train_scaled, y_train)
clf = GBC().fit(X_train_scaled, y_train)
print('Default')
score_tr, score_te = printAccuracy(clf, X_train_scaled, X_test_scaled)
print('Tuned')
score_tr, score_te = printAccuracy(clf_t, X_train_scaled, X_test_scaled)



"""
#RF sens.
bootstrap=False ,max_depth=40,max_features=100,min_samples leaf=1,min_samples_split=15,n_estimators=200,
#SVC acc
#SVC sens
C=4,gamma=0.01,k=all ,shrinking=1,tol=0.001,
#GBC acc
#GBC sens
learning rate=0.01,loss='exponential',max_depth=50,max_features=100,min_samples_leaf=1,min_samples_split=15,n_estimators=600,subsample=0.75
"""