import pandas as pd
import numpy as np
import pickle
from itertools import izip
from random import random
from tools import *
from xgboost import XGBClassifier



date_path = '../input/train_date.csv'
num_path = '../input/train_numeric.csv'


# sample dataset randomly
date_df, num_df = sample(0.05, date_path, num_path)
y_all = pd.read_csv(num_path, index_col=0, usecols=[0,969], dtype=np.float32)


print('\ndetermining important features...')


dataset = pd.concat([date_df, num_df], axis=1)
print(dataset.shape)

colnames = dataset.columns.values

y = y_all.loc[dataset.index].values.ravel()
X = dataset.values

clf = XGBClassifier(base_score=0.005)

print("training...")
clf.fit(X, y)

important_feat_indices = np.where(clf.feature_importances_>0.004)[0] # 0.005
important_feat_names = [colnames[i] for i in important_feat_indices]

print("{} important features\n".format(len(important_feat_names)))

print('writing results to disk...')
with open('good_features_all', 'w') as clf_file:
	clf_file.write(pickle.dumps(important_feat_names))