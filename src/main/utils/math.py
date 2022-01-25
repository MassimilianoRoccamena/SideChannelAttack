import numpy as np

BYTE_SIZE = 256
BYTE_HW_LEN = 9

def kahan_sum(sum_, c, element):
    assert sum_.shape == c.shape
    assert element.shape == c.shape
        
    y= element - c
    t= sum_ + y
    c= (t- sum_) - y
    sum_= t
    return sum_, c

def pca_transform(pca, X):
    #return np.matmul(X-pca.mean_[:X.shape[-1]], pca.components_[:,:X.shape[-1]].T)
    return np.matmul(X-pca.mean_, pca.components_.T)