import pandas as pd
import numpy as np
from operator import itemgetter
import pickle
from StringIO import StringIO as sio
from itertools import izip
from random import random
from tools import *
from xgboost import XGBClassifier
from collections import Counter
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('cluster', metavar='CLUSTER', type=int)


comb_path = 'id-comb-res.csv'

date_path = '../input/train_date.csv'
num_path = '../input/train_numeric.csv'


clusterer_path = 'clusterer'
with open(clusterer_path, 'r') as clus_file:
	clusterer = pickle.loads(clus_file.read())

args = parser.parse_args()

num_clus = len(clusterer.cluster_centers_)

special_clus = args.cluster
# other_clus = np.arange(num_clus)[np.arange(num_clus) != special_clus]

print "cluster = {}".format(special_clus)
# print "other clusters  = {}".format(other_clus)


## CLUSTERING #######################################################
comb_df = pd.read_csv(comb_path, index_col = 0, usecols=[0,1])

print "\nclustering..."
comb_df['combination'] = comb_df['combination'].apply(vectorize)
comb_df['cluster'] = clusterer.predict(np.vstack(comb_df['combination'].values))
comb_df['cluster'] = np.int32(comb_df['cluster'] != special_clus)

clus_dict = comb_df['cluster'].to_dict()
#####################################################################





def sample_uniform(n_rows, cluster_dict):
	# determine rates of sampling for each cluster
	row_counts = [tup[1] for tup in cluster_dict.iteritems()]
	row_counts = Counter(row_counts)
	# row_counts = sorted(row_counts.iteritems(), key=itemgetter(0))
	# row_counts = [rowc for (cluster, rowc) in row_counts]

	n_clusters = len(row_counts)
	total_rows = len(cluster_dict)

	# rates = [n_rows / (1.*rowc) for rowc in row_counts]
	rates = {cluster: n_rows / (1.*row_counts[cluster]) for cluster in row_counts}

	print "Number of clusters: {}".format(n_clusters)
	print "Total number of rows: {}".format(total_rows)
	print "row counts: {}".format(row_counts)
	print "rates: {}".format(rates)

	# create buffers for each cluster (date_buffer, num_buffer)
	buffers = {cluster: (sio(), sio()) for cluster in rates}

	with open(date_path, 'r') as datefile, open(num_path, 'r') as numfile:

		# write headers
		datehead = datefile.readline()
		numhead = numfile.readline()
		for cluster, (date_buf, num_buf) in buffers.iteritems():
			date_buf.write(datehead)
			num_buf.write(numhead)

		# reading rows
		nlines = 0
		for dateln, numln in izip(datefile, numfile):
			nlines += 1

			rowid = int(numln.split(',', 1)[0])
			cluster = cluster_dict[rowid]

			rate = rates[cluster]
			if random() < rate:
				buffers[cluster][0].write(dateln)
				buffers[cluster][1].write(numln)

			if nlines % 100000 == 0: print("progress: {}".format(nlines))

		for cluster in buffers:
			buffers[cluster][0].seek(0)
			buffers[cluster][1].seek(0)


		dataframes = {}
		for cluster in buffers:
			date_buf, num_buf = buffers[cluster][0], buffers[cluster][1]

			dataframes[cluster] = (
				pd.read_csv(date_buf, index_col=0, dtype=np.float32),
				pd.read_csv(num_buf, index_col=0, usecols=list(range(969)), dtype=np.float32)
			)
			
	return dataframes
			



# sample dataset uniformly
datasets = sample_uniform(10000, clus_dict)
y_all = pd.read_csv(num_path, index_col=0, usecols=[0,969], dtype=np.float32)

# check sizes
for i, (datedf, numdf) in datasets.iteritems():
	print "cluster {}".format(i)
	print "{}, {}\n".format(datedf.shape, numdf.shape)



# train a separate model on each dataset:
print('\ndetermining important features for cluster {}...'.format(special_clus))

# important_feats = [[] for _ in range(num_clus)]

# for cluster, (datedf, numdf) in datasets.iteritems():
datedf, numdf = datasets[0]

dataset = pd.concat([datedf, numdf], axis=1)
print(dataset.shape)

colnames = dataset.columns.values

y = y_all.loc[dataset.index].values.ravel()
X = dataset.values

clf = XGBClassifier(base_score=0.005)
clf.fit(X, y)
important_feat_indices = np.where(clf.feature_importances_>0.005)[0] # 0.005
important_feat_names = [colnames[i] for i in important_feat_indices]

print("{} important features\n".format(len(important_feat_indices)))

print('writing results to disk...')
with open("good_features_{}".format(special_clus), 'w') as clf_file:
	clf_file.write(pickle.dumps(important_feat_names))