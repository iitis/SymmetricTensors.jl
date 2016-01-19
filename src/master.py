# -*- coding: utf-8 -*-
"""
Demo for testing transforms in hyperspectral data classification.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.backends.backend_pdf import PdfPages


sys.path.append('low_rank_tensor_approx') # probably wants to modify that

from m_hsbase import load_indian_pines
from util import prepare_hsdata, accuracy
from util import visualize_classes, visualize_training
from scipy import io

# ----------------------------------------------------------------------------

def reduce_X_dimension(X, l, u):
    """Reduce number of spectral features."""
    return X[:,l:u]

# ----------------------------------------------------------------------------

class TransformNone(object):
    def __str__(self):
        return "Nothing"
    
    def fit(self, X):
        pass
    
    def transform(self, X):
        return X

# ----------------------------------------------------------------------------

class TransformPCA(object):
    def __init__(self, k2):
        self.k2 = k2

    def __str__(self):
        return "PCA, n_c = {}".format( self.k2)
    
    def fit(self, X):
        self.pca = RandomizedPCA(5).fit(X)
    
    def transform(self, X):
        return self.pca.transform(X)

# ----------------------------------------------------------------------------

class TransformCumulants(object):

    def __init__(self, k2, k3, k4):
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4

    def __str__(self):
        return "Cumulants k_3 = {}, k_4 = {}, n_c = {}".format(self.k3, self.k4, self.k2)

    def fit(self, X):
        np.save('test.npy', X)

        def run_jm(dir, k3, k4):
            import subprocess
            import matlab.engine
            subprocess.call(["julia", "get_cu.jl"])
            eng = matlab.engine.start_matlab()
            eng.addpath(dir)
            U3, U4 = eng.lrtd(k3, k4, nargout=2)
            return np.mat(U3), np.mat(U4)
        def pca(x, n):
            from scipy import cov, linalg
            y = cov(np.transpose(x))
            [E,U]=linalg.eigh(y)
            sorted_indices_desc = E.real.argsort()[::-1] #first sort, then reverse
            return np.asmatrix(U[:, sorted_indices_desc])


        directory = '/home/krzysztof/Dokumenty/badania_iitis/tensors_symetric/tensor calculations/pictures_tensor/'
        self.U3, self.U4 = run_jm(directory,self.k3,self.k4)
        #self.pca = RandomizedPCA(4).fit(X)
        self.U2 = pca(X,self.k2)

    def transform(self, X):
        features_v = np.hstack([X*self.U2, X*self.U3,X*self.U4])
        print(features_v)
        return features_v
        #return np.hstack([self.pca.transform(X), X*self.U4])
        #return np.hstack([X*self.U3,X*self.U4])
        #return X*self.U3

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    hsdata = load_indian_pines('low_rank_tensor_approx/m_hsbase/data/')
    hsdata = prepare_hsdata(hsdata)
    X, y, idx = hsdata['X'], hsdata['y'], hsdata['train1all']
    l = 70
    u = 130
    X = reduce_X_dimension(X, l, u) # if needed, reduce X dimension

    # display selection of training data and ground truth
    plt.figure()
    visualize_training(hsdata)
    plt.gcf().canvas.set_window_title('Training/test data')
    plt.figure()
    plt.imshow(hsdata['truth'], interpolation='nearest', cmap=plt.cm.spectral)
    plt.gcf().canvas.set_window_title('Ground truth')
    k2_pca = 10
    k2 = 10
    k3 = 2
    k4 = 2

    pp = PdfPages('par_'+str(k2_pca)+'_'+str(k4)+'_'+str(k3)+'_'+str(k2)+'_dat'+str(l)+'_'+str(u)+'.pdf') #KD added line
    # test several transforms
    n_neighbors = 3
    for tt in [TransformNone(), TransformPCA(k2_pca), TransformCumulants(k2,k3,k4)]:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        X_train, y_train = X[idx], y[idx]
        tt.fit(X_train) #zachowuje U1 U2 U3 - stworzyl je
        # moge zapisac X do pliku, wywolac z Julii, zapisac, wywolac z MATLABA i zapisac
        #wywolac proces w Pythonie z Juyli
        Z_train = tt.transform(X_train)  #dostaje X a zwraca U1'X + U2'X + U3'X
        knn.fit(Z_train, y_train)
        Z = tt.transform(X)
        hsdata['yp'] = knn.predict(Z)
        accuracy(hsdata)
        plot = plt.figure()
        visualize_classes(hsdata)
        info = 'Classifier (KNN, n={})'.format(n_neighbors)
        info += ', transform={}'.format(str(tt))
        plt.gcf().canvas.set_window_title(info)
        plt.text(0,157,str(info), fontsize = 10) #KD added line
        pp.savefig(plot) #KD added line
    pp.close() #KD added line
    plt.show()