#!/usr/bin/env python
# coding: utf-8

# In[48]:


from __future__ import print_function

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

plot_3d1 = []
print("Testing Random Forests")
for max_depth in (1, 2, 3, 4, 5): 
	for n_estimators in [10, 25, 50, 100, 200]:
		for min_samples_split in [2, 3, 4, 5]:
			for min_samples_leaf in [0.5, 1, 2, 4, 8]:
				print(max_depth, n_estimators, min_samples_split, min_samples_leaf, end= ' ')
				clf = RF(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf).fit(X_train, y_train)
				score_tr, score_te = printAccuracy(clf, X_train, X_test)
				if score_te > best_score:
					best_score = score_te
					best_rf = clone(clf)
                    
				#acc = printAccuracy(clf, X_train, X_test)
				plot_3d1.append([n_estimators, max_depth, best_rf])                    
                    
plot_3d1 = np.array(plot_3d1)
from mpl_toolkits.mplot3d import Axes3D 

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')

pltX1, pltY1 = plot_3d1[:,0], plot_3d1[:,1]
pltZ1 = plot_3d1[:,-1]

#df = pd.DataFrame({'x': pltX, 'y': pltY, 'z': pltZ}, index=range(len(pltX)))

surf1 = ax1.plot_trisurf(df.x, df.y, df.z, cmap='jet', linewidth=0.1, edgecolor='k')
ax1.set_xlabel('Max_depth')
ax1.set_ylabel('N estimators')
ax1.set_zlabel('Test accuracy')
fig1.colorbar(surf1, shrink=0.5, aspect=5)

plt.show()
                    
                    
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.svm import SVC

best_score = float('-inf')

plot_3d2 = []
print("\nTesting SVM")
for c in [0.125,0.25, 0.5, 1, 2, 4, 8, 16]:
	for g in [1, 2, 3, 4, 5, 'auto']:
		for tol in [0.001, 0.006, 0.011, 0.016]:
			for coef0 in [0, 0.01, 0.02, 0.03]:
				clf = SVC(C=c, tol = tol, coef0 = coef0, kernel = 'rbf', gamma = g).fit(X_train_scaled, y_train)
				print(c, g, tol, coef0, end= ' ')
				score_tr, score_te = printAccuracy(clf, X_train_scaled, X_test_scaled)
				if score_te > best_score:
					best_score = score_te
					best_svm = clone(clf)

					plot_3d2.append([c, tol, best_svm])  
                    
plot_3d2 = np.array(plot_3d2)
from mpl_toolkits.mplot3d import Axes3D 

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')

pltX2, pltY2 = plot_3d2[:,0], plot_3d2[:,1]
pltZ2 = plot_3d2[:,-1]

surf2 = ax2.plot_trisurf(df.x, df.y, df.z, cmap='jet', linewidth=0.1, edgecolor='k')
ax2.set_xlabel('C')
ax2.set_ylabel('Tol')
ax2.set_zlabel('Test Accuracy')
fig2.colorbar(surf2, shrink=0.5, aspect=5)

plt.show(2)
                                        

    
from sklearn.ensemble import GradientBoostingClassifier as GBC

best_score = float('-inf')

plot_3d3 = []
print("\nTesting GBC")
for learning_rate in [0.0125, 0.025, 0.05, 0.1, 0.5, 1 ]:
	for n_estimators in [10, 25, 50, 100, 200]:
		for max_depth in [1, 2, 3, 4, 5]:
			for max_leaf_nodes in [2, 5, 10, 20, 25]:
				clf = GBC(learning_rate = learning_rate, n_estimators = n_estimators, max_depth = max_depth, max_leaf_nodes = max_leaf_nodes ).fit(X_train_scaled, y_train)
				print(learning_rate, n_estimators, max_depth, max_leaf_nodes, end= ' ')
				score_tr, score_te = printAccuracy(clf, X_train_scaled, X_test_scaled)
				if score_te > best_score:
					best_score = score_te
					best_gbc = clone(clf)

					plot_3d3.append([learning_rate, n_estimators, best_gbc]) 
                    
plot_3d3 = np.array(plot_3d3)
from mpl_toolkits.mplot3d import Axes3D 

fig = plt.figure()
ax = fig.gca(projection='3d')

pltX, pltY = plot_3d3[:,0], plot_3d3[:,1]
pltZ = plot_3d3[:,-1]

surf3 = ax.plot_trisurf(df.x, df.y, df.z, cmap='jet', linewidth=0.1, edgecolor='k')
ax.set_xlabel('Learning Rate')
ax.set_ylabel('N estimators')
ax.set_zlabel('Test Accuracy')
fig.colorbar(surf3, shrink=0.5, aspect=5)

plt.show()                    
                    
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




