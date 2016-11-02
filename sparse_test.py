from tools import to_sparse, to_dense_csc, load_sparse, load_ohe

import numpy as np
import pandas as pd

from time import sleep


# path = '../input/test.csv'

# df = pd.read_csv(path, index_col=0)

# print df.values
# print

# header, mtx = to_sparse(df)

# print to_dense(mtx)
# print header


# path = '../input/test_numeric_sparse'

# header, mtx = load_sparse(path)

# print header.shape
# print mtx.shape
# print mtx.__class__
# sleep(20)


# header, mtx = load_ohe('../input/train_cat_sparse')
# print 'loaded'
# sleep(20)


header, num_mtx = load_sparse('../input/test_numeric_sparse')
num_df = pd.read_csv('../input/test_numeric.csv', index_col=0, nrows=100)
df_cols = num_df.columns.values

print len(df_cols)
print len(header)


usecols = header[680:694]
mtx_subset = num_mtx[:25, 680:694]

df_subset = num_df.iloc[:25][usecols].values
mtx_subset = np.around(to_dense_csc(mtx_subset), 3)


print "\nDIFF:"
print np.around(df_subset - mtx_subset, 8)
