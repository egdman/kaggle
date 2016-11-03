from operator import itemgetter
import pickle
from argparse import ArgumentParser

# parser = ArgumentParser()
# parser.add_argument('cluster', metavar='CLUSTER_NUM', type=int)

# args = parser.parse_args()
# clus = args.cluster

feats_path = 'features'

with open(feats_path, 'r') as feat_file:
	important_feats = pickle.loads(feat_file.read())



for clus, ifeats in enumerate(important_feats):
	print "cluster {}".format(clus)
	stations = set()
	for ifeat in ifeats:
		stations.add(ifeat.split('_')[1][1:] + '-' + ifeat.split('_')[1])

	print sorted(stations)
	print