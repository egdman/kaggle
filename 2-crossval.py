import pandas as pd
import numpy as np
import gc
import pickle
from xgboost import XGBClassifier, DMatrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from argparse import ArgumentParser

from tools import load_sparse, get_corr_mtx, mcc_eval, eval_mcc, load_ohe, write_csv

from matplotlib import pyplot as plt
from operator import itemgetter
from math import floor
import scipy.sparse as spar




def get_important_indices(header, important_colnames):
	present_colanmes = np.intersect1d(header, important_colnames)
	col_indices = np.arange(header.shape[0])[np.in1d(header, present_colanmes)]
	return col_indices



def mcc_eval_invert(y_prob, dtrain):
	name, score = mcc_eval(y_prob, dtrain)
	return name, -score




num_path = '../input/train_numeric_sparse'
date_path = '../input/train_date_sparse'
cat_path = '../input/train_cat_sparse'

feat_path = 'important_feats_all.csv'

ifeats = pd.read_csv(feat_path, index_col=0, dtype=str).index.values

feat_cutoff = 80
ifeats = ifeats[:feat_cutoff]



######## DATE ###################################
dhead, dmtx = load_sparse(date_path)

d_colids = get_important_indices(dhead, ifeats)
print("{} date features".format(len(d_colids)))

dmtx = dmtx[:, d_colids]
print(dmtx.shape)
gc.collect()




######## NUMERIC ################################
nhead, nmtx = load_sparse(num_path)

corr_mtx = get_corr_mtx(nhead, nmtx)

y = nmtx[:, -1]
y = np.array(y.todense()).ravel()

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
print("y :  {}".format(y.shape))

y_pred = np.empty(y.shape)


print('\ncross validating...')

cv = StratifiedKFold(n_splits=3, shuffle=True)


############## XGB PARAMETERS ##############
xgb_params = {
	'max_depth': 5,
	'n_estimators' : 100,
	'learning_rate': 0.05,  # 0.15
	'base_score': 0.005,
	'colsample_bytree': 0.9
}
############################################

mcc_scores = []

# for learning_rate in learning_rates:
for i, (train, test) in enumerate(cv.split(X, y)):
	print("\n{} fold:".format(i))

	clf = XGBClassifier(**xgb_params)

	print(clf)

	clf.fit(X[train], y[train],
		# eval_metric=mcc_eval_invert,
		# eval_set = [(X[test], y[test])],
		# early_stopping_rounds=10 # maybe
	)

	y_pred[test] = clf.predict_proba(X[test])[:,1]
	fold_mcc = eval_mcc(y[test], y_pred[test])

	print("ROC AUC = {:.3f}".format(roc_auc_score(y[test], y_pred[test])))
	print("MCC     = {:.3f}".format(fold_mcc))

print('')

final_auc = roc_auc_score(y, y_pred)
final_mcc = eval_mcc(y, y_pred)

print('------------------------------------------------------')
print("FINAL AUC = {:.3f}".format(final_auc))
print("FINAL MCC = {:.3f}".format(final_mcc))
print('------------------------------------------------------')

# mcc_scores.append(final_mcc)


# resfname = 'lrates-{}-{}.csv'.format(
# 	np.around(min(learning_rates), 3),
# 	np.around(max(learning_rates), 3))


# with open(resfname, 'w') as result_file:
# 	write_csv(result_file, [['learning_rate', 'mcc']])
# 	write_csv(result_file, zip(learning_rates, mcc_scores))



#### pick the best threshold out-of-fold ################################################
thresholds = np.linspace(0.01, 0.99, 50)
mcc = np.array([matthews_corrcoef(y, np.int32(y_pred>thr)) for thr in thresholds])

plt.plot(thresholds, mcc)
plt.ylim((0, 0.4))

best_threshold = thresholds[mcc.argmax()]
best_mcc = mcc.max()

print("\nbest MCC: {}".format(best_mcc))
print("best threshold: {}".format(best_threshold))

plt_fname = "date-num-corr-{}_feats.png".format(feat_cutoff)
plt.savefig(plt_fname)
plt.clf()




# #### COMMENT THIS OUT TO PREVENT TRAINING OF THE FINAL CLASSIFIER #######################
# #### FINAL TRAINING #####################################################################
# train, test = train_test_split(np.arange(X.shape[0]), test_size = 0.1)
# print("Fitting on {} rows".format(len(train)))



# final_clf = XGBClassifier(**xgb_params)
# final_clf.fit(X, y)
# # final_clf.fit(X[train], y[train],
# # 	eval_metric=mcc_eval_invert,
# # 	eval_set = [(X[test], y[test])],
# # 	early_stopping_rounds=10
# # )

# result_fname = "{}_thr={}_MCC={}".format(feat_cutoff, best_threshold, best_mcc)
# clf_fname = 'predictor_' + result_fname

# print("writing predictor to: %s" % clf_fname)
# with open(clf_fname, 'w') as resfile:
# 	resfile.write(pickle.dumps(final_clf))
# #########################################################################################