import pandas as pd
import numpy as np
import gc
import pickle
from xgboost import XGBClassifier, DMatrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from argparse import ArgumentParser

from tools import load_sparse, mcc_eval, eval_mcc, get_ifeats, get_corr_mtx

from matplotlib import pyplot as plt
from operator import itemgetter
from math import floor
import scipy.sparse as spar


def mcc_eval_invert(y_prob, dtrain):
	name, score = mcc_eval(y_prob, dtrain)
	return name, -score


num_path = '../input/train_numeric_sparse'




# with open('good_num_features_all', 'r') as featfile:
# 	ifeats = pickle.loads(featfile.read())
num_feats = 400

ifeats, importances = get_ifeats('num_feature_importances.csv')
print "top feature: {} (importance = {})".format(ifeats[0], importances[0])
ifeats = ifeats[0:num_feats]


print "{} important numeric features".format(len(ifeats))

print "loading numeric features..."

header, mtx = load_sparse(num_path)


## WITH SUBSAMPLING ######################################################
FRAC = 0.6 # .23 # subsampling fraction
index = np.arange(mtx.shape[0])
sub_size = int(mtx.shape[0] * FRAC)

sub_index = np.random.choice(index, size=sub_size, replace=False)
sub_mtx = mtx[sub_index] # SUBSAMPLING
# sub_mtx = mtx            # NO SUBSAMPLING

del mtx
gc.collect()


X = sub_mtx[:, ifeats]
y = sub_mtx[:, 968]

corr_mtx = get_corr_mtx(header, sub_mtx)
corr_mtx = spar.csc_matrix(corr_mtx) # convert to sparse matrix so that we can stack it with X
print("corr mtx: {}".format(corr_mtx.shape))

del sub_mtx
gc.collect()

# add correlation features to X
X = spar.hstack([X, corr_mtx])
# X = corr_mtx
##########################################################################


## WITHOUT SUBSAMPLING ###################################################
# X = mtx[:, ifeats]
# y = mtx[:, 968]
# del mtx
# gc.collect()
##########################################################################


y = np.array(y.todense()).ravel()

print("X: {}".format(X.shape))
print("y: {}".format(y.shape))


print("\ncross validating...")

y_pred = np.empty(y.shape)


################ PARAMETERS ################
max_depth = 3
n_est = 1000
learning_rate = 0.03
############################################

classifiers = []

cv = StratifiedKFold(n_splits=3, shuffle=True)

for i, (train, test) in enumerate(cv.split(X, y)):
	print("\n{} fold:".format(i))
	clf = XGBClassifier(max_depth = max_depth, n_estimators = n_est, learning_rate=learning_rate, base_score=0.005)

	clf.fit(X[train], y[train],
		eval_metric=mcc_eval_invert,
		eval_set = [(X[test], y[test])],
		early_stopping_rounds=10 # maybe
	)

	# y_pred[test] = clf.fit(X[train], y[train]).predict_proba(X[test])[:,1]
	y_pred[test] = clf.predict_proba(X[test])[:,1]

	fold_mcc = eval_mcc(y[test], y_pred[test])
	print("ROC AUC = {:.3f}".format(roc_auc_score(y[test], y_pred[test])))

	print("MCC     = {:.3f}".format(fold_mcc))
	# classifiers.append((clf, fold_mcc))

print     
print("FINAL SCORE = {:.3f}".format(roc_auc_score(y, y_pred)))


#### pick the best threshold out-of-fold ################################################
thresholds = np.linspace(0.01, 0.99, 50)
mcc = np.array([matthews_corrcoef(y, np.int32(y_pred>thr)) for thr in thresholds])

plt.plot(thresholds, mcc)
plt.ylim((0, 0.4))

best_threshold = thresholds[mcc.argmax()]
best_mcc = mcc.max()

print("best MCC: {}".format(best_mcc))
print("best threshold: {}".format(best_threshold))



# best_clf, best_mcc = sorted(classifiers, key=itemgetter(1), reverse=True)[0]
# print("best MCC*: {}".format(best_mcc))

result_fname = "{}_thr={}_MCC={}".format(num_feats, best_threshold, best_mcc)
clf_fname = 'predictor_' + result_fname
plt_fname = 'mccplot_' + result_fname + '.png'

plt.savefig(plt_fname)
plt.clf()
#########################################################################################

# with open(clf_fname, 'w') as resfile:
# 	resfile.write(pickle.dumps(best_clf))



# #### FINAL TRAINING #####################################################################
# train, test = train_test_split(np.arange(X.shape[0]), test_size = 0.1)
# print("Fitting on {} rows".format(len(train)))


# final_clf = XGBClassifier(max_depth = max_depth, n_estimators = n_est, learning_rate=learning_rate, base_score=0.005)
# final_clf.fit(X[train], y[train],
# 	eval_metric=mcc_eval_invert,
# 	eval_set = [(X[test], y[test])],
# 	early_stopping_rounds=10
# )


# print("writing predictor to: %s" % clf_fname)
# with open(clf_fname, 'w') as resfile:
# 	resfile.write(pickle.dumps(final_clf))
# #########################################################################################