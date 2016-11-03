import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from operator import itemgetter
from matplotlib import pyplot as plt
import pickle
from argparse import ArgumentParser

from tools import *
from StringIO import StringIO


good_center = ('0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25'
				',26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51\n'
	          '-0.0,-0.0,0.0,-0.0,-0.0,0.0,-0.0,0.0,-0.0,0.0,0.0,0.0,0.0,0.0,-0.0'
	          	',0.0,0.0,0.0,0.01,0.01,0.03,0.02,0.01,0.01,0.47,0.39,0.33,0.32,0.15,0.77'
	          	',0.83,0.02,0.46,0.71,0.78,0.46,0.48,0.9,0.27,0.0,0.0,-0.0,0.0,0.0,-0.0,-0.0'
	          	',-0.0,0.01,0.01,0.0,0.0,0.01')




parser = ArgumentParser()
parser.add_argument('-test', action='store_true')
parser.add_argument('num_clusters', metavar='NUM_CLUSTERS', type=int)

args = parser.parse_args()
num_clusters = args.num_clusters
test = args.test
word = 'test' if test else 'train'


comb_path = 'enumerated_combinations_%s.csv' % word

good_center = pd.read_csv(StringIO(good_center)).loc[0].values

uniq_combinations = (pd
	.read_csv(comb_path, usecols=["combination"])['combination']
	.apply(vectorize))

# select random points as initial centers of clusters
rnd_centers_ids = np.random.choice(
	uniq_combinations.index.values,
	size=(num_clusters-1),
	replace=False)

init_centers = np.vstack([
	np.vstack(uniq_combinations.loc[rnd_centers_ids].values),
	good_center])

uniq_combinations = np.vstack(uniq_combinations.values)


print init_centers.shape
print uniq_combinations.shape


kmeans = KMeans(n_clusters = num_clusters, init = init_centers)
kmeans.fit(uniq_combinations)

# let's draw clustered combinations
# create vector column of cluster labels
labels_col = np.array(kmeans.labels_)[:, np.newaxis]

# add 1 to every label so there is no zeros
labels_col += 1

labeled_vectors = np.multiply(uniq_combinations, labels_col)

# sort by label:
lab_vec = [(lab[0], vec) for lab, vec in zip(labels_col, labeled_vectors)]
labeled_vectors = [vec for lab, vec in sorted(lab_vec, key = itemgetter(0))]


# write the cluster model to disk
with open('clusterer', 'w') as resfile:
	resfile.write(pickle.dumps(kmeans))


# plot clusters
fig = plt.figure(figsize = (14, 7))
ax = fig.add_subplot(111)
ax.matshow(labeled_vectors, aspect='auto')
fig.savefig("combinations.png")

# save cluster centers
centers = kmeans.cluster_centers_

centers_df = pd.DataFrame(data = np.around(centers[:], 2))

centers_df.to_csv('centers')
# print centers_df



