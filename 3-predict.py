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


def get_numfeats(fname):
	return int(re.search(r'(?<=predictor_)(\d*)', fname).group(0))

def get_thresh(fname):
	return float(re.search(r'(?<=thr\=)(0\.\d*)', fname).group(0))



parser = ArgumentParser()
parser.add_argument('predictor', metavar='PREDICTOR', type=str)
parser.add_argument('-t', '--threshold', metavar='THRESHOLD', type=float, default = -100.)

args = parser.parse_args()



id_path  = '../input/test_numeric.csv'
num_path = '../input/test_numeric_sparse'
date_path = '../input/test_date_sparse'
cat_path = '../input/test_cat_sparse'
feat_path = 'important_feats_all.csv'
predictor_path = args.predictor


#### PREDICTOR NAME PARSING TO GET PARAMETERS ##################
predictor_name = path.basename(predictor_path)
num_feat = get_numfeats(predictor_name)

if args.threshold < 0:
	threshold = get_thresh(predictor_name)
else:
	threshold = args.threshold

print("%d features" % num_feat)
print('threshold = %s' % threshold)



ifeats = pd.read_csv(feat_path, index_col=0, dtype=str).index.values
ifeats = ifeats[:num_feat]



######## NUMERIC ################################
nhead, nmtx = load_sparse(num_path)

corr_mtx = get_corr_mtx(nhead, nmtx)

n_colids = get_important_indices(nhead, ifeats)
print("{} numeric features".format(len(n_colids)))

nmtx = nmtx[:, n_colids]
print(nmtx.shape)
gc.collect()




######## CORR ###################################
corr_mtx = spar.csc_matrix(corr_mtx)
corr_head = np.array( list(str(i) for i in np.arange(corr_mtx.shape[1])) )

corr_colids = get_important_indices(corr_head, ifeats)
print("{} correlation features".format(len(corr_colids)))

corr_mtx = corr_mtx[:, corr_colids]
print(corr_mtx.shape)




######## CAT ####################################
_, cat_mtx = load_ohe(cat_path)
cat_head = np.array( list('CAT' + str(i) for i in range(cat_mtx.shape[1])) )

cat_colids = get_important_indices(cat_head, ifeats)
print("{} categorical features".format(len(cat_colids)))

cat_mtx = cat_mtx[:, cat_colids].tocsc()
print(cat_mtx.shape)



############# CONSTRUCT FEATURE MTX #############
X = spar.hstack([dmtx, nmtx, corr_mtx, cat_mtx])
#################################################

print("X :  {}".format(X.shape))

print("\npredicting...")

with open(predictor_path, 'r') as predfile:
	clf = pickle.loads(predfile.read())



y_proba = clf.predict_proba(X)[:,1]
y_labels = np.int32(y_proba > threshold)

# y_labels = np.int32(clf.predict(X))
	
print("writing predictions to disk...")

pred_df = pd.read_csv(id_path, index_col=0, usecols=[0], dtype=np.int32)
pred_df['Response'] = y_labels

result_fname = predictor_name + '_submission'
pred_df.to_csv(result_fname)