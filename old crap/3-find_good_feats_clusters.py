import pandas as pd
import numpy as np
from operator import itemgetter
import pickle
from StringIO import StringIO as sio
from itertools import izip
from random import random
from tools import *
from xgboost import XGBClassifier

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-n', '--num-samples', type=int, default = 10000)
parser.add_argument('cluster_labels', metavar='CLUSTER_LABELS', nargs='*', type=int)

args = parser.parse_args()

req_labels = args.cluster_labels


comb_path = 'id-comb-train.csv'

date_path = '../input/train_date.csv'
num_path = '../input/train_numeric.csv'


clusterer_path = 'clusterer'
with open(clusterer_path, 'r') as clus_file:
	clusterer = pickle.loads(clus_file.read())



## CLUSTERING #######################################################
# num_clus = len(clusterer.cluster_centers_)
comb_df = pd.read_csv(comb_path, index_col = 0, usecols=[0,1])

print "\nclustering..."

# fast, high memory
comb_df['combination'] = comb_df['combination'].apply(vectorize)
comb_df['cluster'] = clusterer.predict(np.vstack(comb_df['combination'].values))

# # slow, low memory
# comb_df['cluster'] = comb_df['combination'].apply(lambda x: clusterer.predict([vectorize(x)])[0])

clus_dict = comb_df['cluster'].to_dict()
#####################################################################


# leave only requested labels:
if len(req_labels) > 0:
	clus_dict = {Id: label for Id, label in clus_dict.iteritems() if label in req_labels}



# sample dataset uniformly
datasets = sample_uniform(args.num_samples, clus_dict, date_path, num_path)
y_all = pd.read_csv(num_path, index_col=0, usecols=[0,969], dtype=np.float32)

# check sizes
for i, (datedf, numdf) in datasets.iteritems():
	print "cluster {}".format(i)
	print "{}, {}\n".format(datedf.shape, numdf.shape)



# train a separate model on each dataset:
print('\ndetermining important features...')

for label, (datedf, numdf) in datasets.iteritems():
	print("cluster #{}".format(label))

	dataset = pd.concat([datedf, numdf], axis=1)
	print(dataset.shape)

	colnames = dataset.columns.values

	y = y_all.loc[dataset.index].values.ravel()
	X = dataset.values

	clf = XGBClassifier(base_score=0.005)
	clf.fit(X, y)
	important_feat_indices = np.where(clf.feature_importances_>0.01)[0] # 0.005
	important_feat_names = list(colnames[i] for i in important_feat_indices)

	print("{} important features\n".format(len(important_feat_names)))

	with open("good_features_{}".format(label), 'w') as resfile:
		resfile.write(pickle.dumps(important_feat_names))