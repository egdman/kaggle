import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from argparse import ArgumentParser

from tools import *

from matplotlib import pyplot as plt
from itertools import chain
from operator import itemgetter


parser = ArgumentParser()
parser.add_argument('cluster_labels', metavar='CLUSTER_LABELS', nargs='*', type=int)

args = parser.parse_args()
req_labels = args.cluster_labels

NROWS = 100000

date_path = '../input/train_date.csv'
num_path = '../input/train_numeric.csv'

comb_path = 'id-comb-train.csv'
clusterer_path = 'clusterer'



with open(clusterer_path, 'r') as clus_file:
	clusterer = pickle.loads(clus_file.read())



if len(req_labels) == 0:
	req_labels = range(len(clusterer.cluster_centers_))


# get important features
ifeats = {}
for reql in req_labels:
	feats_path = "good_features_{}".format(reql)
	with open(feats_path, 'r') as feat_file:
		ifeats[reql] = pickle.loads(feat_file.read())



# get column names for both input files
num_cols = pd.read_csv(num_path, index_col=0, nrows=1).drop(['Response'], axis=1).columns.values
date_cols = pd.read_csv(date_path, index_col=0, nrows=1).columns.values



# find union of important features for all clusters
sets = list(set(lof) for lof in ifeats.values())
ifeat_union = list(set.union(*sets))
print "{} important features in total".format(len(ifeat_union))



# figure out what features belong to which files
date_feats = np.intersect1d(date_cols, ifeat_union)
num_feats = np.intersect1d(num_cols, ifeat_union)



## CLUSTERING #######################################################
comb_df = pd.read_csv(comb_path, index_col = 0, usecols=[0,1])

print "\nclustering..."
comb_df['combination'] = comb_df['combination'].apply(vectorize)
comb_df['cluster'] = clusterer.predict(np.vstack(comb_df['combination'].values))


# set 'cluster' column as index
clus_dict = (comb_df
		.drop(['combination'], axis=1)
		.reset_index()
		.set_index('cluster'))
#####################################################################



print "\nreading data..."
# read the full set
full_X = pd.concat([
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

full_y = pd.read_csv(num_path, index_col=0, dtype=np.float32, usecols=[0,969])



print("\ncross-validating...")
# crossvalidate
for (label, feat_list) in sorted(ifeats.iteritems(), key=itemgetter(0)):
	print("{} important features for {}-th cluster".format(len(feat_list), label))

	# get all Ids of the requested cluster
	clus_ids = clus_dict.loc[label, 'Id']
	print("found {} examples of cluster {}".format(clus_ids.shape, label))

	cluster_X = full_X.loc[clus_ids, feat_list]
	cluster_y = full_y.loc[cluster_X.index]

	print "cluster set: {}".format(cluster_X.shape)

	y = cluster_y.values.ravel()
	y_pred = np.ones(y.shape[0])
	X = cluster_X.values

	clf = XGBClassifier(max_depth=5, base_score=0.005)
	cv = StratifiedKFold(y, n_folds=3)

	try:
		for i, (train, test) in enumerate(cv):
			y_pred[test] = clf.fit(X[train], y[train]).predict_proba(X[test])[:,1]
			print("fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(y[test], y_pred[test])))

		print("FINAL SCORE: {}".format(roc_auc_score(y, y_pred)))

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
	print
	# word = 'not_' if args.other_than else ''
	plt.savefig("mcc_{}.png".format(label))