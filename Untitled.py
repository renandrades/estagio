#!/usr/bin/env python
# coding: utf-8

# In[22]:


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


def plotCorrMap(ini, end):
	corr = data[data.columns[ini:end+1]].corr()
	mask = np.zeros_like(corr)
	mask[np.triu_indices_from(mask)] = True

	ax = sns.heatmap(corr.abs(), annot=True, fmt = '.1f', mask = mask,linewidths=.5)
	plt.tight_layout()
	#plt.savefig('teste.pdf')(salvar figura)
	plt.show()


def plotCorrWithClass(ini, end, class_name):
	corr = data[data.columns[ini:end+1]].corrwith(data[class_name])
	corr.abs().plot(kind='bar')
	plt.tight_layout()
	plt.show()

data1 = pd.read_csv('data_arrhythmia.csv', sep = ';', na_values='?')
data = data1.drop(data1[(data1['height'] > 250)].index) #retirei os valores invalidos de altura (700m)

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

print ('\nBanco com %d amostras e %d colunas\n' % data.shape)
#data.target.shape

print ("Tabela sumarizando as colunas do banco\n")
print (data.describe())

X = data.loc[:, data.columns != 'diagnosis']
y = data['diagnosis']

from sklearn.ensemble import RandomForestClassifier as RF 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
#from joblib import dump, load    (No module named joblib)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

best_score = float('-inf')

"""
file_rf = open("Tuning_rf_acc.csv","w")

plot_3d = []
print("Testing Random Forests")
for max_depth in (1, 2, 3, 4, 5): 
	for n_estimators in [10, 25, 50, 100, 200]:
		for max_features in [2, 4, 6, 8, 10, 15, 25, 50, "auto"]:
			for min_samples_split in [2, 4, 8, 16, 32]:
				print(max_depth, n_estimators, max_features, min_samples_split, end= ' ')
				clf = RF(max_depth = max_depth, n_estimators = n_estimators, max_features = max_features, min_samples_split = min_samples_split).fit(X_train, y_train)
				score_tr, score_te = printAccuracy(clf, X_train, X_test)
				if score_te > best_score:
					best_score = score_te
					best_rf = clone(clf)
                    
				#acc = printAccuracy(clf, X_train, X_test)
				plot_3d.append([n_estimators, max_depth, score_te])                    
				rf_cv = cross_val_score(clf, X_train, y_train, cv=5)
				print (max_depth, n_estimators, max_features, min_samples_split, rf_cv.mean(), score_te, file=file_rf)
                
dump( best_rf, 'best_rf.pkl')     

                
plot_3d = np.array(plot_3d)
from mpl_toolkits.mplot3d import Axes3D 

fig = plt.figure()
ax = fig.gca(projection='3d')

pltX, pltY = plot_3d[:,0], plot_3d[:,1]
pltZ = plot_3d[:,-1]

df = pd.DataFrame({'x': pltX, 'y': pltY, 'z': pltZ}, index=range(len(pltX)))

surf = ax.plot_trisurf(df.x, df.y, df.z, cmap='jet', linewidth=0.1, edgecolor='k')
ax.set_xlabel('N estimators')
ax.set_ylabel('max_depth')
ax.set_zlabel('Test accuracy')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
file_rf.close()
"""    
    
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.svm import SVC

best_score = float('-inf')

file_svm = open("Tuning_svm_acc.csv","w")

#plot_3d = []
print("\nTesting SVM")
for c in [0.125,0.25, 0.5, 1, 2, 4, 8, 16]:
	for g in [0.0625, 0.125, 0.25, 0.5, 2, 4, 8, 16, 32, 'auto']:
		for tol in [0.001, 0.006, 0.011, 0.016]:
			clf = SVC(C=c, gamma = g, tol = tol, kernel = 'rbf').fit(X_train_scaled, y_train)
			print(c, g, tol, end= ' ')
			score_tr, score_te = printAccuracy(clf, X_train_scaled, X_test_scaled)
			if score_te > best_score:
				best_score = score_te
				best_svm = clone(clf)
                    
			#acc = printAccuracy(clf, X_train, X_test)
			#plot_3d.append([c, g, score_te])
			svm_cv = cross_val_score(clf, X_train_scaled, y_train, cv=5)                
			print (c, g, tol, svm_cv.mean(), score_te, file=file_svm)

dump(best_svm, 'best_svm.pkl')            

"""
plot_3d = np.array(plot_3d)
from mpl_toolkits.mplot3d import Axes3D 

fig = plt.figure()
ax = fig.gca(projection='3d')

pltX, pltY = plot_3d[:,0], plot_3d[:,1]
pltZ = plot_3d[:,-1]

df = pd.DataFrame({'x': pltX, 'y': pltY, 'z': pltZ}, index=range(len(pltX)))

surf = ax.plot_trisurf(df.x, df.y, df.z, cmap='jet', linewidth=0.1, edgecolor='k')
ax.set_xlabel('C')
ax.set_ylabel('Tol')
ax.set_zlabel('Test accuracy')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
"""
file_svm.close()


from sklearn.ensemble import GradientBoostingClassifier as GBC

best_score = float('-inf')

file_gbc = open("Tuning_gbc_acc.csv","w")

plot_3d = []
print("\nTesting GBC")
for learning_rate in [0.0125, 0.025, 0.05, 0.1, 0.5, 1 ]:
	for n_estimators in [10, 25, 50, 100, 200]:
		for max_depth in [1, 2, 3, 4, 5]:
			clf = GBC(learning_rate = learning_rate, n_estimators = n_estimators, max_depth = max_depth).fit(X_train_scaled, y_train)
			print(learning_rate, n_estimators, max_depth, end= ' ')
			score_tr, score_te = printAccuracy(clf, X_train_scaled, X_test_scaled)
			if score_te > best_score:
				best_score = score_te
				best_gbc = clone(clf)

			#acc = printAccuracy(clf, X_train, X_test)
			plot_3d.append([learning_rate, n_estimators, score_te])
			gbc_cv = cross_val_score(clf, X_train_scaled, y_train, cv=5)
			print (learning_rate, n_estimators, max_depth, gbc_cv.mean(), score_te, file=file_gbc)

dump(best_gbc, 'best_gbc.pkl')                
                    
plot_3d = np.array(plot_3d)
from mpl_toolkits.mplot3d import Axes3D 

fig = plt.figure()
ax = fig.gca(projection='3d')

pltX, pltY = plot_3d[:,0], plot_3d[:,1]
pltZ = plot_3d[:,-1]

df = pd.DataFrame({'x': pltX, 'y': pltY, 'z': pltZ}, index=range(len(pltX)))

surf = ax.plot_trisurf(df.x, df.y, df.z, cmap='jet', linewidth=0.1, edgecolor='k')
ax.set_xlabel('Learning rate')
ax.set_ylabel('N estimators')
ax.set_zlabel('Test accuracy')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
file_gbc.close()                    

for col in data.columns[:14]:

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


# In[ ]:





# In[ ]:





# In[ ]:




