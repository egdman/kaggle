import pandas as pd
from argparse import ArgumentParser
import numpy as np
from tools import vectorize
from os import path
import pickle

parser = ArgumentParser()
parser.add_argument('input', metavar='FILE', type=str)


args = parser.parse_args()
input_path = args.input

comb_path = 'id-comb-train.csv'
clusterer_path = 'kmeans.model'

with open(clusterer_path, 'r') as clus_file:
	clusterer = pickle.loads(clus_file.read())


input_df = pd.read_csv(input_path, index_col=['Id'])


## CLUSTERING #######################################################
comb_df = pd.read_csv(comb_path, index_col = 0, usecols=[0,1])

# drop excess Ids:
comb_df = comb_df.loc[input_df.index]

print "\nclustering..."
comb_df['combination'] = comb_df['combination'].apply(vectorize)
comb_df['cluster'] = clusterer.predict(np.vstack(comb_df['combination'].values))


# set 'cluster' column as index
clus_dict = (comb_df
		.drop(['combination'], axis=1)
		.reset_index()
		.set_index('cluster'))
#####################################################################


all_clusters = range(len(clusterer.cluster_centers_))

for clus in all_clusters:
	clus_ids = clus_dict.loc[clus, 'Id']
	print "cluster {}: {} rows".format(clus, clus_ids.shape[0])
	out_df = input_df.loc[clus_ids]
	out_path = path.join(path.dirname(input_path), path.basename(input_path) + "_cluster_{}".format(clus))
	print "writing result to {}".format(out_path)
	out_df.to_csv(out_path)


