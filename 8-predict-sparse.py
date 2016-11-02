import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from argparse import ArgumentParser
from os import path
import scipy.sparse as spar

from tools import *

from matplotlib import pyplot as plt
from itertools import chain
import gc
import re


def get_numfeats(fname):
	return int(re.search(r'(?<=predictor_)(\d*)', fname).group(0))

def get_thresh(fname):
	return float(re.search(r'(?<=thr\=)(0\.\d*)', fname).group(0))



parser = ArgumentParser()
parser.add_argument('predictor', metavar='PREDICTOR', type=str)
parser.add_argument('-t', '--threshold', metavar='THRESHOLD', type=float, default = -100.)

args = parser.parse_args()
NROWS = 10000


num_path = '../input/test_numeric_sparse'
id_path  = '../input/test_numeric.csv'
predictor_path = args.predictor

predictor_name = path.basename(predictor_path)

num_feat = get_numfeats(predictor_name)

if args.threshold < 0:
	threshold = get_thresh(predictor_name)
else:
	threshold = args.threshold


print("%d features" % num_feat)
print('threshold = %s' % threshold)


ifeats, importances = get_ifeats('num_feature_importances.csv')
ifeats = ifeats[0:num_feat]

print("{} important numeric features".format(len(ifeats)))

y_df = pd.read_csv(id_path, index_col=0, usecols=[0], dtype=np.int32)

print ("loading data...")
header, mtx = load_sparse(num_path)

corr_mtx = get_corr_mtx(header, mtx)

X = mtx[:, ifeats]

del mtx
gc.collect()


corr_mtx = spar.csc_matrix(corr_mtx)
X = spar.hstack([X, corr_mtx])


print("\npredicting...")

with open(predictor_path, 'r') as predfile:
	clf = pickle.loads(predfile.read())

print "X: {}".format(X.shape)

y_proba = clf.predict_proba(X)[:,1]
y_labels = np.int32(y_proba > threshold)

# y_labels = np.int32(clf.predict(X))
	
print("writing predictions to disk...")

y_df['Response'] = y_labels

result_fname = predictor_name + '_submission'
y_df.to_csv(result_fname)


