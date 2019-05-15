import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def plotCorrMap(ini, end):
	corr = data[data.columns[ini:end+1]].corr()
	mask = np.zeros_like(corr)
	mask[np.triu_indices_from(mask)] = True

	ax = sns.heatmap(corr.abs(), annot=True, fmt = '.1f', mask = mask,linewidths=.5)
	plt.tight_layout()
	plt.show()


def plotCorrWithClass(ini, end, class_name):
	corr = data[data.columns[ini:end+1]].corrwith(data[class_name])
	corr.abs().plot(kind='bar')
	plt.tight_layout()
	plt.show()

data = pd.read_csv('data_arrhythmia.csv', sep = ';', na_values='?')

for col in data.columns:
	num_null = data[col].isnull().sum()
	if num_null:
		print('Column %s has %d (%.1f%%) null values' % (col, num_null, 100.0*num_null/data.shape[0]))

for col in data.columns:
	num_unique = np.unique(data[col])
	if len(num_unique) == 1:
		print('Column %s has a single value and will be discarded' % (col))
		data.drop(columns=[col], inplace = True)
#data.diagnosis.value_counts().div(data.shape[0]/100.0).plot(kind='barh')
#plt.show()
data.fillna(data.mean(), inplace = True)

data.loc[data.diagnosis == 1, 'diagnosis'] = 0
data.loc[data.diagnosis  > 1, 'diagnosis'] = 1

X = data.loc[:, data.columns != 'diagnosis']
y = data['diagnosis']

from sklearn.ensemble import RandomForestClassifier as RF 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

best_score = float('-inf')
print("Testing Random Forests")
for max_d in range(1,10):
	#continue
	
	print(max_d,end= ' ')
	clf = RF(max_depth=max_d, n_estimators = 100).fit(X_train, y_train)
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
for c in [0.125,0.25, 0.5, 1, 2, 4, 8, 16]:
	#continue
	clf = SVC(C=c, kernel = 'rbf', gamma = 'auto').fit(X_train_scaled, y_train)
	print(c,end= ' ')
	score_tr, score_te = printAccuracy(clf, X_train_scaled, X_test_scaled)
	if score_te > best_score:
		best_score = score_te
		best_svm = clone(clf)


from sklearn.ensemble import GradientBoostingClassifier as GBC

best_score = float('-inf')
print("\nTesting GBC")
for learning_rate in [0.0125, 0.025, 0.05, 0.1, 0.5, 1 ]:
	#continue
	for max_features in [1,2,3,4,5]:
		clf = GBC(learning_rate = learning_rate, max_features =max_features).fit(X_train_scaled, y_train)
		print(learning_rate, max_features,end= ' ')
		score_tr, score_te = printAccuracy(clf, X_train_scaled, X_test_scaled)
		if score_te > best_score:
			best_score = score_te
			best_gbc = clone(clf)


for col in data.columns[:14]:
	#Scontinue

	unique = np.unique(data[col])
	if len(unique) > 50:
		hist0 = data.loc[data.diagnosis == 0, col].plot.density(alpha = 0.5, label = 'NEG', color = 'blue')
		hist1 = data.loc[data.diagnosis == 1, col].plot.density(alpha = 0.5, label = 'POS', color = 'red')
	else:
		hist0 = data.loc[data.diagnosis == 0, col].plot.hist(alpha = 0.5, label = 'NEG', color = 'blue')
		hist1 = data.loc[data.diagnosis == 1, col].plot.hist(alpha = 0.5, label = 'POS', color = 'red')
	plt.xlabel(col)
	plt.legend()
	plt.show()



plotCorrMap(0,14)
plotCorrWithClass(0,14, 'diagnosis')

plotCorrMap(15,26)
plotCorrWithClass(15,26, 'diagnosis')









