# -*- coding: utf-8 -*-
"""
Demo for testing transforms in hyperspectral data classification.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import RandomizedPCA, NMF
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.backends.backend_pdf import PdfPages


sys.path.append('low_rank_tensor_approx') # probably wants to modify that

from m_hsbase import load_indian_pines
from util import prepare_hsdata, accuracy
from util import visualize_classes, visualize_training
from scipy import io
from joblib import Memory
mem = Memory(cachedir='/tmp/joblib')

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
    def __init__(self, k2_pca):
        self.k2_pca = k2_pca

    def __str__(self):
        return "PCA, n_c = {}".format( self.k2_pca)
    
    def fit(self, X):
        self.pca = RandomizedPCA(k2_pca).fit(X)
    
    def transform(self, X):
        return self.pca.transform(X)

# ----------------------------------------------------------------------------

class TransformNMF(object):
    def __init__(self, k2_nf):
        self.k2_nf = k2_nf

    def __str__(self):
        return "NMF, n_c = {}".format( self.k2_nf)

    def fit(self, X):
        self.nmf = NMF(self.k2_nf).fit(X)

    def transform(self, X):
        return self.nmf.transform(X)

# ----------------------------------------------------------------------------


class TransformPhi(object):

    def __init__(self, k2):
        self.k2 = k2


    def __str__(self):
        return "Cumulants_common k = {}".format(self.k2)

    def fit(self, X):
        @mem.cache
        def cumulants(X):
            import subprocess
            np.save('test.npy', X)
            subprocess.call(["julia", "cumulants_only.jl"])

        def pca_save(x, n):
            from scipy import cov, linalg
            y = cov(np.transpose(x))
            [E,U]=linalg.eigh(y)
            sorted_indices_desc = E.real.argsort()[::-1]
            U2 = np.asmatrix(U[:, sorted_indices_desc][:,0:n])
            np.save('U2.npy', U2)


        cumulants(X)
        np.save('parameter.npy', k2)
        #pca_save(X, k2)
        import subprocess
        subprocess.call(["julia", "ALS.jl"])

    def transform(self, X):

        U_c = np.mat(np.load("U_common.npy"))
        print(np.size(U_c,0))
        features_v = X*U_c
        return features_v


# ----------------------------------------------------------------------------



if __name__ == '__main__':
    hsdata = load_indian_pines('low_rank_tensor_approx/m_hsbase/data/')
    hsdata = prepare_hsdata(hsdata)
    X, y, idx = hsdata['X'], hsdata['y'], hsdata['train1all']
    l = 0
    u = 130
    X = reduce_X_dimension(X, l, u) # if needed, reduce X dimension oryginalne dane
    #np.save('X.npy', X)


    # display selection of training data and ground truth
    plt.figure()
    visualize_training(hsdata)
    plt.gcf().canvas.set_window_title('Training/test data')
    plt.figure()
    plt.imshow(hsdata['truth'], interpolation='nearest', cmap=plt.cm.spectral)
    plt.gcf().canvas.set_window_title('Ground truth')
    k = 8

    k2_pca = k
    k2 = k
    k2_nf = k

    pp = PdfPages('par_'+str(k2_pca)+'_'+str(k2_nf)+'_'+str(k2)+'_dat'+str(l)+'_'+str(u)+'.pdf') #KD added line
    # test several transforms
    n_neighbors = 3
    for tt in [TransformNone(), TransformPCA(k2_pca), TransformPhi(k2), TransformNMF(k2_nf)]:
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