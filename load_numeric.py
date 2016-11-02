import pandas as pd
from scipy.sparse import csr_matrix, hstack
import numpy as np
from tools import load_sparse, to_dense
import time
import pickle

train_path = 'train_numeric_sparse'


tr_header, tr_mtx = load_sparse(train_path)


print "TRAIN: {}".format(tr_mtx.shape)
