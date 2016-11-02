import pandas as pd
import numpy as np
import gc
import pickle
from xgboost import XGBClassifier, DMatrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from argparse import ArgumentParser

from tools import load_sparse, get_corr_mtx, write_csv, sample_mtx_rows, load_ohe

from matplotlib import pyplot as plt
from operator import itemgetter
from math import floor
import scipy.sparse as spar


def get_names_by_index(index, *name_arrays):
	return list(np.concatenate(name_arrays)[i] for i in index)



############## XGB PARAMETERS ##############
max_depth = 3
n_est = 1000
learning_rate = 0.03
############################################



num_path = '../input/train_numeric_sparse'
date_path = '../input/train_date_sparse'
cat_path = '../input/train_cat_sparse'

sample_frac = 0.25



######## NUMERIC ################################
nhead, nmtx = load_sparse(num_path)
nmtx = sample_mtx_rows(nmtx, sample_frac)
print(nmtx.shape)
gc.collect()

y = nmtx[:, -1]
y = np.array(y.todense()).ravel()

nmtx = nmtx[:, :-1] # remove the response column (the last one)
nhead = nhead[:-1] # remove the 'Response' column (the last one)



######## DATE ###################################
dhead, dmtx = load_sparse(date_path)
dmtx = sample_mtx_rows(dmtx, sample_frac)
print(dmtx.shape)
gc.collect()



######## CORR ###################################
corr_mtx = get_corr_mtx(nhead, nmtx)
corr_mtx = spar.csc_matrix(corr_mtx)
corr_head = np.arange(corr_mtx.shape[1])
print(corr_mtx.shape)



######## CAT ####################################
_, cat_mtx = load_ohe(cat_path)
cat_mtx = sample_mtx_rows(cat_mtx, sample_frac).tocsc()
print(cat_mtx.shape)
gc.collect()

# we generate own categorical header instead of the one provided because it has repeating column names
cat_head = np.array( list('CAT' + str(i) for i in range(cat_mtx.shape[1])) )



############# CONSTRUCT FEATURE MTX #############
X = spar.hstack([dmtx, nmtx, corr_mtx, cat_mtx])
#################################################

print("X :  {}".format(X.shape))

print('\ndetermining important features...')
clf = XGBClassifier(
	max_depth = max_depth,
	n_estimators = n_est,
	learning_rate=learning_rate,
	base_score=0.005)


clf.fit(X, y)

sorted_feats = np.argsort(clf.feature_importances_)[::-1] # sorted from most to least important

feat_names = get_names_by_index(sorted_feats, dhead, nhead, corr_head, cat_head)
feat_names = np.array([feat_names]).T

print "writing results..."
with open('important_feats_all.csv', 'w') as featfile:
	write_csv(featfile, feat_names)