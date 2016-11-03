import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import gc

from tools import load_sparse, mcc_eval, get_ifeats


def mcc_eval_invert(y_prob, dtrain):
	name, score = mcc_eval(y_prob, dtrain)
	return name, -score


NROWS = 100000

num_path = '../input/train_numeric_sparse'

# feats_path = 'good_num_features_all'
# with open(feats_path, 'r') as feat_file:
# 	ifeats = np.array(pickle.loads(feat_file.read()))


ifeats, importances = get_ifeats('num_feature_importances.csv')
ifeats = ifeats[0:160]


print("{} important numeric features".format(len(ifeats)))


header, mtx = load_sparse(num_path)

X = mtx[:, ifeats]
y = mtx[:, 968]
y = np.array(y.todense()).ravel()

del mtx
gc.collect()

# crossvalidate a model for the requested cluster
print("\nfitting...")

clf = XGBClassifier(max_depth=5, n_estimators = 100, base_score=0.005)

print "X:      {}".format(X.shape)
print "y:      {}".format(y.shape)


# clf.fit(X, y)

train, test = train_test_split(np.arange(X.shape[0]), test_size = 0.1)

clf.fit(X[train], y[train],
	eval_metric=mcc_eval_invert,
	eval_set = [(X[test], y[test])],
	early_stopping_rounds=20 # maybe
)
	
print("writing predictor to disk...")
with open('predictor_sparse_160num', 'w') as resfile:
	resfile.write(pickle.dumps(clf))