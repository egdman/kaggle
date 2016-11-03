import pandas as pd
from scipy.sparse import csr_matrix, hstack
import numpy as np
from tools import load_ohe
import time
import pickle

train_path = '../input/train_cat_sparse'
test_path = '../input/test_cat_sparse'


tr_header, tr_mtx = load_ohe(train_path)
te_header, te_mtx = load_ohe(test_path)


print "TRAIN: {}".format(tr_mtx.shape)
print " TEST: {}".format(te_mtx.shape)