import pandas as pd
import numpy as np
import pickle
from itertools import izip
from random import random
from tools import *
from xgboost import XGBClassifier

num_path = '../input/train_numeric_sparse'



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

print('\ndetermining important features...')


clf = XGBClassifier(base_score=0.005)

print("training...")
clf.fit(X, y)

important_feat_indices = np.where(clf.feature_importances_>0.002)[0] # 0.005

print("{} important features\n".format(len(important_feat_indices)))


print('writing results to disk...')
with open('good_num_features_all', 'w') as resfile:
	resfile.write(pickle.dumps(important_feat_indices))