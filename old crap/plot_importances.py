import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


data = pd.read_csv(
	'num_feature_importances.csv',
	index_col='feat_index',
	usecols=['feat_index', 'final_score']
)

feat_ids = data.index.values
scores = data.values
print scores.shape



plt.figure(figsize=(12,8))
for i in range(scores.shape[1]):
	plt.plot(feat_ids, scores[:,i])


plt.show()
