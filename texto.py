#coding: utf-8

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", message="UserWarning")
warnings.warn = lambda *a, **kw: False

import pandas as pd

# sep = ";" indica que o separador do banco de dados #Nesse caso é separado por vírgula
Banco4 = pd.read_csv("./NovosTestes/Banco4.csv", sep = ";" ) 



Banco4.loc[Banco4['Classe'] == 1, 'Classe'] = 0
Banco4.loc[Banco4['Classe'] > 1, 'Classe'] = 1






import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

train_cols = Banco4.columns[:-1]
X = Banco4[train_cols]
y = Banco4['Classe']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
X_train = X_train.loc[:, ["QRS (ms)", "T (ms)"]]

X_test = X_test.loc[:, ["QRS (ms)", "T (ms)"]]

h = .02  # step size in the mesh

#names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
 #        "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
  #       "Naive Bayes", "QDA"]

names = ["RBF SVM", "Decision Tree", "Random Forest", "Neural Net"]


classifiers = [
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),]


figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets


    # just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1, len(classifiers) + 1, i)

ax.set_title("Input data")


    # Plot the training points
ax.scatter(X_train[ 'QRS (ms)'], X_train['T (ms)'], c=y_train, cmap=cm_bright, edgecolors='k')
    # Plot the testing points
ax.scatter(X_test['QRS (ms)'], X_test['T (ms)'], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')

x_min, x_max =  X[ 'QRS (ms)'].min() - .5, X[ 'QRS (ms)'].max() + .5
y_min, y_max =  X['T (ms)'].min() - .5, X['T (ms)'].max() + .5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))


ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i += 1

    # iterate over classifiers
for name, clf in zip(names, classifiers):
    #print 'Rnu
    ax = plt.subplot(1, len(classifiers) + 1, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot the training points
    ax.scatter(X_train[ 'QRS (ms)'], X_train[ 'T (ms)'], c=y_train, cmap=cm_bright, edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[ 'QRS (ms)'], X_test[ 'T (ms)'], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
    i += 1

plt.tight_layout()
plt.show()
