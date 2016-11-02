import pandas as pd
import numpy as np
import pickle
from itertools import izip
from random import random
from tools import *
from xgboost import XGBClassifier


num_path = '../input/train_numeric_sparse'

num_rounds = 20

all_scores = []

for rd in range(num_rounds):
	print "\nROUND {}".format(rd)

	header, mtx = load_sparse(num_path)

	index = np.arange(mtx.shape[0])
	sub_index = np.random.choice(index, size=60000, replace=False)

	sub_mtx = mtx[sub_index]
	del mtx
	X = sub_mtx[:, :968]
	y = sub_mtx[:, 968]
	y = np.array(y.todense()).ravel()

	print "X: {}".format(X.shape)
	print "y: {}".format(y.shape)

	print('determining important features...')
	clf = XGBClassifier(base_score=0.005)
	clf.fit(X, y)

	feat_scores = np.array(clf.feature_importances_, ndmin=2)
	all_scores.append(feat_scores)



scores = np.concatenate(all_scores, axis=0)
final_scores = np.array([scores.mean(axis=0)])
feat_indices = np.array([range(scores.shape[1])])

data = np.concatenate((feat_indices, scores, final_scores)).T

columns = list("round_{}".format(rd) for rd in range(num_rounds))
columns = ["feat_index"] + columns + ["final_score"]




print data.shape

with open("feature_importances.csv", 'w') as resfile:
	write_csv(resfile, [columns])
	write_csv(resfile, data)