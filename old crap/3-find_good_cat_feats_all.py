import pandas as pd
import numpy as np
import pickle
from itertools import izip
from random import random
from tools import *
from xgboost import XGBClassifier


num_path = '../input/train_numeric.csv'
cat_path = '../input/train_cat_sparse'


header, mtx = load_ohe(cat_path)

index = np.arange(mtx.shape[0])
sub_index = np.random.choice(index, size=100000, replace=False)

labels = pd.read_csv(num_path, usecols=[969], dtype=np.float32)

X = mtx[sub_index]
y = labels.loc[sub_index].values.ravel()

print "X: {}".format(X.shape)
print "y: {}".format(y.shape)

print('\ndetermining important features...')


clf = XGBClassifier(base_score=0.005)

print("training...")
clf.fit(X, y)

important_feat_indices = np.where(clf.feature_importances_>0.012)[0] # 0.005

print("{} important features\n".format(len(important_feat_indices)))


print('writing results to disk...')
with open('good_cat_features_all', 'w') as resfile:
	resfile.write(pickle.dumps(important_feat_indices))