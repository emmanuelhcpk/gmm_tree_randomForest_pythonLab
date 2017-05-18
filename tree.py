
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from sklearn.datasets import load_iris
from sklearn import tree
import scipy
from scipy import stats # para normalizacion
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy.matlib

mat = scipy.io.loadmat('Data.mat')
datos =  mat['Data']
X=datos[:310,1:6]
y=datos[:310,6:7]
folds = 20 # repeticiones

# se crea el modelo
clf = tree.DecisionTreeClassifier(max_depth =15)
EficienciaTest = np.zeros(folds)
for fold in range(0,folds) : # se realiza numero de repeticiones cross validation
    kf = KFold(n_splits=folds)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        xtrain, xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
    #normalizacion
    xtrain = normalize(xtrain)
    mu = np.mean(xtrain)
    sigma = np.std(xtrain)
    aux1=np.matlib.repmat(mu,len(xtest),1)
    aux2=np.matlib.repmat(sigma,len(xtest),1)
    resta = xtest-aux1
    xtest = np.divide(resta,aux2)
    yest = np.zeros(len(ytest))
    #print xtest
    clf = clf.fit(xtrain, ytrain)
    for i in range(len(ytest)): #validacion
        yest[i] = clf.predict([xtest[i]])
    # se genera la matriz de confusion
    confusion = confusion_matrix(ytest, yest)
    #calculo la eficiencia
    EficienciaTest[fold]=np.divide(np.sum(np.diag(confusion)),np.sum(np.sum(confusion)))
Eficiencia = np.mean(EficienciaTest)
std = np.std(EficienciaTest)
print Eficiencia
print std