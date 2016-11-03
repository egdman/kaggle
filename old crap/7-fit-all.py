import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from argparse import ArgumentParser

from tools import *

from matplotlib import pyplot as plt
from itertools import chain


NROWS = 100000

date_path = '../input/train_date.csv'
num_path = '../input/train_numeric.csv'
cat_path = '../input/train_cat_sparse'


feats_path = 'good_features_all'
cat_feats_path = 'good_cat_features_all'

with open(feats_path, 'r') as feat_file:
	ifeats = np.array(pickle.loads(feat_file.read()))

with open(cat_feats_path, 'r') as cat_feat_file:
	cat_ifeats = np.array(pickle.loads(cat_feat_file.read()))


print("{} important date and num features".format(len(ifeats)))
print("{} important categorical  features".format(len(cat_ifeats)))


# figure out what features belong to which files
num_cols = pd.read_csv(num_path, index_col=0, nrows=1).drop(['Response'], axis=1).columns.values
date_cols = pd.read_csv(date_path, index_col=0, nrows=1).columns.values

date_feats = np.intersect1d(date_cols, ifeats)
num_feats = np.intersect1d(num_cols, ifeats)

print("important numeric features: {}".format(len(num_feats)))
print("important date features:    {}".format(len(date_feats)))

# # load numeric and date features:
# datenum_set = pd.concat([
# 	pd.read_csv(
# 		date_path,
# 		index_col=0,
# 		usecols=np.concatenate([['Id'], date_feats]),
# 		# nrows=NROWS
# 	),
# 	pd.read_csv(
# 		num_path,
# 		index_col=0,
# 		usecols=np.concatenate([['Id'], num_feats]),
# 		# nrows=NROWS
# 	)
# 	], axis=1)


# load numeric features only:
datenum_set = pd.read_csv(
	num_path,
	index_col=0,
	usecols=np.concatenate([['Id'], num_feats]))



# print "loading cat feats"
# cat_header, cat_mtx = load_ohe(cat_path)
# cat_mtx = cat_mtx[:, cat_ifeats].todense()

# print "cat set: {}".format(cat_mtx.shape)
print "datenum set: {}".format(datenum_set.shape)


y = pd.read_csv(num_path, index_col=0, dtype=np.float32, usecols=[0,969])
y = y.loc[datenum_set.index]


# crossvalidate a model for the requested cluster
print("\nfitting...")


y = y.values.ravel()
# X = np.hstack([datenum_set.values, cat_mtx])
X = datenum_set.values

clf = XGBClassifier(max_depth=5, n_estimators = 100, base_score=0.005)

print "X:      {}".format(X.shape)
print "y:      {}".format(y.shape)


clf.fit(X, y)
	
print("writing predictor to disk...")
with open('predictor_dense', 'w') as resfile:
	resfile.write(pickle.dumps(clf))

