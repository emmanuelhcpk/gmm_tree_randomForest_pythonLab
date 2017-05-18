import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
folds = 35 # repeticiones

# se crea el modelo random forest
numero_arboles = 500
rf = RandomForestClassifier(n_estimators=numero_arboles,max_depth =20)

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
    aux1 = np.matlib.repmat(mu, len(xtest), 1)
    aux2 = np.matlib.repmat(sigma, len(xtest), 1)
    resta = xtest - aux1
    xtest = np.divide(resta,aux2)
    yest = np.zeros(len(ytest))
    rf = rf.fit(xtrain, ravel(ytrain))
    for i in range(len(ytest)): #validacion
        yest[i] = rf.predict([xtest[i]])
    confusion = confusion_matrix(ytest, yest)
    #calculo la eficiencia
    EficienciaTest[fold]=(np.sum(np.diag(confusion))/np.sum(np.sum(confusion)))
Eficiencia = np.mean(EficienciaTest)
std = np.std(EficienciaTest)
print Eficiencia
print std