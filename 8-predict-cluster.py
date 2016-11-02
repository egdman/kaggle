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


parser = ArgumentParser()
parser.add_argument('cluster', metavar='CLUSTER_NUM', type=int)
parser.add_argument('threshold', metavar='THRESHOLD', type=float)

args = parser.parse_args()
clus = args.cluster

NROWS = 100000

# date_path = '../input/test_date.csv'
# num_path = '../input/test_numeric.csv'

date_path = '../input/train_date.csv'
num_path = '../input/train_numeric.csv'

comb_path = 'id-comb-res.csv'
clusterer_path = 'kmeans.model'
feats_path = "good_features_{}".format(clus)
predictor_path = "predictor_{}".format(clus)


with open(feats_path, 'r') as feat_file:
	ifeats = np.array(pickle.loads(feat_file.read()))

with open(clusterer_path, 'r') as clus_file:
	clusterer = pickle.loads(clus_file.read())

print("{} important features for {}-th cluster".format(len(ifeats), clus))





# figure out what features belong to which files
num_cols = pd.read_csv(num_path, index_col=0, usecols=list(range(969)), nrows=1).columns.values
date_cols = pd.read_csv(date_path, index_col=0, nrows=1).columns.values
print("numeric features: {}".format(len(num_cols)))
print("date features:    {}".format(len(date_cols)))

date_feats = np.intersect1d(date_cols, ifeats)
num_feats = np.intersect1d(num_cols, ifeats)

print("important numeric features: {}".format(len(num_feats)))
print("important date features:    {}".format(len(date_feats)))



## CLUSTERING #######################################################
num_clus = len(clusterer.cluster_centers_)
comb_df = pd.read_csv(comb_path, index_col = 0, usecols=[0,1])

print "\nclustering..."
comb_df['combination'] = comb_df['combination'].apply(vectorize)
comb_df['cluster'] = clusterer.predict(np.vstack(comb_df['combination'].values))


# set 'cluster' column as index
clus_dict = (comb_df
		.drop(['combination'], axis=1)
		.reset_index()
		.set_index('cluster'))

# get all Ids of the requested cluster
clus_ids = clus_dict.loc[clus, 'Id']
print("found {} examples of cluster {}".format(clus_ids.shape, clus))
#####################################################################


full_set = pd.concat([
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

full_set = full_set.loc[clus_ids]


print "\nfull set: {}".format(full_set.shape)


print("\npredicting...")

X = full_set.values

with open(predictor_path, 'r') as predfile:
	clf = pickle.loads(predfile.read())


print "X: {}".format(X.shape)

y_proba = clf.predict_proba(X)[:,1]

y_labels = np.int32(y_proba > args.threshold)
	
print("writing predictions to disk...")

y_df = pd.DataFrame(data=y_labels, index=full_set.index, columns=['Response'])

y_df.to_csv("test_predictions_{}.csv".format(clus))

