
from sklearn.mixture import GaussianMixture
import numpy.matlib
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

mat = scipy.io.loadmat('Data.mat')
datos =  mat['Data']
X=datos[:310,1:6]
y=datos[:310,6:7]
folds = 20 # repeticiones
# separar por clases
Xc1 = []
Xc2 = []
Xc3 = []
Yc1 = []
Yc2 = []
Yc3 = []
for i in range(len(X)) :
     if (y[i] == 1):
         Xc1.append(X[i])
         Yc1.append(y[i])
     elif (y[i] == 2):
         Xc2.append(X[i])
         Yc2.append(y[i])
     elif (y[i] == 3):
         Xc3.append(X[i])
         Yc3.append(y[i])

#print Xc1
# se crea el modelo
EficienciaTest = np.zeros(folds)
EficienciaTest2 = np.zeros(folds)
EficienciaTest3 = np.zeros(folds)
n_components_range = range(1, 6) # numero de componentes
cv_types = ['spherical', 'tied', 'diag', 'full']  # tipos de matrices de covarianza

for range in n_components_range:
    for cv in cv_types:
        gmm = GaussianMixture(n_components=range, covariance_type=cv) # creo el modelo
        for fold in range(1,folds) : # se realiza numero de repeticiones cross validation
            kf = KFold(n_splits=folds)
            # clase 1
            kf.get_n_splits(Xc1)
            for train_index, test_index in kf.split(Xc1):
                xtrain1, xtest1 = X[train_index], X[test_index]
                ytrain1, ytest1 = y[train_index], y[test_index]
            #clase 2
            kf.get_n_splits(Xc2)
            for train_index, test_index in kf.split(Xc2):
                xtrain2, xtest2 = X[train_index], X[test_index]
                ytrain2, ytest2 = y[train_index], y[test_index]
            #clase 3
            kf.get_n_splits(Xc3)
            for train_index, test_index in kf.split(Xc3):
                xtrain3, xtest3 = X[train_index], X[test_index]
                ytrain3, ytest3 = y[train_index], y[test_index]
            #normalizacion
            xtrain1 = normalize(xtrain1)
            mu = np.mean(xtrain1)
            sigma = np.std(xtrain1)
            aux1 = np.matlib.repmat(mu, len(xtest1), 1)
            aux2 = np.matlib.repmat(sigma, len(xtest1), 1)
            resta = xtest1 - aux1
            xtest1 = resta / aux2
            yest1 = np.zeros(len(ytest1))

            xtrain2 = normalize(xtrain2)
            mu = np.mean(xtrain2)
            sigma = np.std(xtrain2)
            aux1 = np.matlib.repmat(mu, len(xtest2), 1)
            aux2 = np.matlib.repmat(sigma, len(xtest2), 1)
            resta = xtest2 - aux1
            xtest2 = resta / aux2
            yest2 = np.zeros(len(ytest2))

            xtrain3 = normalize(xtrain3)
            mu = np.mean(xtrain3)
            sigma = np.std(xtrain3)
            aux1 = np.matlib.repmat(mu, len(xtest3), 1)
            aux2 = np.matlib.repmat(sigma, len(xtest3), 1)
            resta = xtest3 - aux1
            xtest3 = resta / aux2
            yest3 = np.zeros(len(ytest3))

            # modelo para cada clase
            gmmC1 = gmm.fit(xtrain1)
            gmmC2 = gmm.fit(xtrain2)
            gmmC3 = gmm.fit(xtrain3)
            for i in range(len(ytest1)): #validacion
                yest1[i] = gmmC1.predict([xtest1[i]])

            for i in range(len(ytest2)): #validacion
                yest2[i] = gmmC2.predict([xtest2[i]])
            # se genera la matriz de confusion
            for i in range(len(ytest3)): #validacion
                yest3[i] = gmmC3.predict([xtest3[i]])
            # se genera la matriz de confusion

            Matriz = [yest1,yest2,yest3]
            confusion = confusion_matrix(ytest1, yest1)

            #calculo la eficiencia
            EficienciaTest[fold]=(np.sum(np.diag(confusion))/np.sum(np.sum(confusion)))

std = np.std(EficienciaTest)
print EficienciaTest
print std