import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from argparse import ArgumentParser

from tools import *

from matplotlib import pyplot as plt
from itertools import chain


NROWS = 200000

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
print("\ncross-validating...")


y = y.values.ravel()
y_pred = np.ones(y.shape[0])

# X = np.hstack([datenum_set.values, cat_mtx])
X = datenum_set.values

print "X:      {}".format(X.shape)
print "y:      {}".format(y.shape)
print "y_pred: {}".format(y_pred.shape)


# max_depths = [5, 6, 7, 8]
# n_ests = [100, 125, 150]

max_depth = 5
n_est = 100


# for max_depth in max_depths:
# 	for n_est in n_ests:

print "\nDEPTH = {}, NEST = {}".format(max_depth, n_est)
clf = XGBClassifier(max_depth=max_depth, n_estimators = n_est, base_score=0.005)
# clf = EnsembleXGB(50, max_depth=5, base_score=0.005)
# clf = EnsembleXGBLogReg(50, max_depth=5, base_score=0.005)
cv = StratifiedKFold(n_splits=3)



try:
	for i, (train, test) in enumerate(cv.split(X, y)):
		y_pred[test] = clf.fit(X[train], y[train]).predict_proba(X[test])[:,1]
		print("fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(y[test], y_pred[test])))

	print("\nFINAL SCORE: {}".format(roc_auc_score(y, y_pred)))

except ValueError as ex:
		print(ex.message)


# pick the best threshold out-of-fold
thresholds = np.linspace(0.01, 0.99, 50)
mcc = np.array([matthews_corrcoef(y, np.int32(y_pred>thr)) for thr in thresholds])

plt.plot(thresholds, mcc)
plt.ylim((0, 0.4))

best_threshold = thresholds[mcc.argmax()]
print("best MCC: {}".format(mcc.max()))
print("best threshold: {}".format(best_threshold))
plt.savefig("mcc_all_maxdepth_{}_nest_{}.png".format(max_depth, n_est))
plt.clf()