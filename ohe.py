import pandas as pd
import scipy.sparse as spar
import numpy as np
from itertools import izip, chain
import pickle
from tools import one_hot, save_ohe, chunker

cat_path = '../input/train_categorical.csv'

chunksize = 30

columns = pd.read_csv(
	cat_path,
	index_col=0,
	nrows = 1,
	dtype=str).columns.values

ncols = len(columns)
print(ncols)


headers = []
matrices = []
progress = 0
# read file in vertical chunks
for col_subset in chunker(columns, chunksize):

	progress += chunksize

	usecols = np.append(['Id'], col_subset)
	df = pd.read_csv(
		cat_path,
		index_col=0,
		dtype=str,
		nrows=10000,
		usecols=usecols
	)

	header, mtx = one_hot(df)
	if header is not None:
		matrices.append(mtx)
		headers.append(header)
		print("progress = {:.1f}% : {}".format(100.*progress / (1.*ncols), mtx.shape))



full_header = np.hstack(headers)
full_mtx = spar.hstack(matrices)

print full_header.shape
print full_mtx.shape


with open('train_cat_sparse_small', 'wb') as resfile:
	save_ohe(resfile, full_header, full_mtx.tocsr())