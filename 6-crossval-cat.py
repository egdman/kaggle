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


# parser = ArgumentParser()
# parser.add_argument('threshold', metavar='THRESHOLD', type=float)
# args = parser.parse_args()


# test_path = '../input/test_cat_sparse'

train_path = '../input/train_cat_sparse'
num_path = '../input/train_numeric.csv'


print "loading cat features"
cat_header, cat_mtx = load_ohe(train_path)


with open('good_cat_features_all', 'r') as featfile:
	ifeats = pickle.loads(featfile.read())


print "{} important categorical feats".format(len(ifeats))

labels = pd.read_csv(num_path, usecols=[969])


print("\ncross validating...")


X = cat_mtx[:, ifeats]
y = labels.values.ravel()
y_pred = np.empty(y.shape)

print("X: {}".format(X.shape))
print("y: {}".format(y.shape))


clf = XGBClassifier(max_depth = 5, n_estimators = 100, base_score=0.005)
cv = StratifiedKFold(n_splits=3)

for i, (train, test) in enumerate(cv.split(X, y)):
	print("\n{} fold:".format(i))

	y_pred[test] = clf.fit(X[train], y[train]).predict_proba(X[test])[:,1]
	print("ROC AUC = {:.3f}".format(roc_auc_score(y[test], y_pred[test])))

print     
print("FINAL SCORE = {:.3f}".format(roc_auc_score(y, y_pred)))


# pick the best threshold out-of-fold
thresholds = np.linspace(0.01, 0.99, 50)
mcc = np.array([matthews_corrcoef(y, np.int32(y_pred>thr)) for thr in thresholds])

plt.plot(thresholds, mcc)
plt.ylim((0, 0.4))

best_threshold = thresholds[mcc.argmax()]
print("best MCC: {}".format(mcc.max()))
print("best threshold: {}".format(best_threshold))
plt.savefig("mcc_cat_subset_100.png")
plt.clf()