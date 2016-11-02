import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from argparse import ArgumentParser

from tools import *

from matplotlib import pyplot as plt
from itertools import chain


parser = ArgumentParser()
parser.add_argument('threshold', metavar='THRESHOLD', type=float)
parser.add_argument('predictor', metavar='PREDICTOR', type=str)

args = parser.parse_args()
NROWS = 10000

date_path = '../input/test_date.csv'
num_path = '../input/test_numeric.csv'
cat_path = '../input/test_cat_sparse'

# date_path = '../input/train_date.csv'
# num_path = '../input/train_numeric.csv'
# cat_path = '../input/train_cat_sparse'

feats_path = 'good_features_all'
cat_feats_path = 'good_cat_features_all'

predictor_path = args.predictor

with open(feats_path, 'r') as feat_file:
	ifeats = np.array(pickle.loads(feat_file.read()))

with open(cat_feats_path, 'r') as cat_feat_file:
	cat_ifeats = np.array(pickle.loads(cat_feat_file.read()))


print("{} important date and num features".format(len(ifeats)))
print("{} important categorical  features".format(len(cat_ifeats)))


# figure out what features belong to which files
num_cols = pd.read_csv(num_path, index_col=0, usecols=list(range(969)), nrows=1).columns.values
date_cols = pd.read_csv(date_path, index_col=0, nrows=1).columns.values

date_feats = np.intersect1d(date_cols, ifeats)
num_feats = np.intersect1d(num_cols, ifeats)

print("important numeric features: {}".format(len(num_feats)))
print("important date features:    {}".format(len(date_feats)))


datenum_set = pd.concat([
	pd.read_csv(
		date_path,
		index_col=0,
		usecols=np.concatenate([['Id'], date_feats]),
		# nrows=NROWS
	),
	pd.read_csv(
		num_path,
		index_col=0,
		usecols=np.concatenate([['Id'], num_feats]),
		# nrows=NROWS
	)
	], axis=1)



print "loading cat feats"
cat_header, cat_mtx = load_ohe(cat_path)
cat_mtx = cat_mtx[:, cat_ifeats].todense()

print "cat set: {}".format(cat_mtx.shape)
print "datenum set: {}".format(datenum_set.shape)


print("\npredicting...")

X = np.hstack((datenum_set.values, cat_mtx))
# X = datenum_set.values

with open(predictor_path, 'r') as predfile:
	clf = pickle.loads(predfile.read())


print "X: {}".format(X.shape)

y_proba = clf.predict_proba(X)[:,1]
y_labels = np.int32(y_proba > args.threshold)

# y_labels = np.int32(clf.predict(X))
	
print("writing predictions to disk...")

y_df = pd.DataFrame(data=y_labels, index=datenum_set.index, columns=['Response'])

y_df.to_csv('test_predictions.csv')
# y_df.to_csv('train_predictions.csv')


