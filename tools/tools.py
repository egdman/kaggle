import numpy as np
import pandas as pd
from random import random
from StringIO import StringIO as sio
from collections import Counter
from itertools import izip, chain
import scipy.sparse as spar
from operator import itemgetter



def vectorize(string):
	vec = np.zeros(52)
	for i, char in enumerate(string):
		vec[i] = int(char)
	return vec



def station_number(col_name):
    return int(col_name.split('_')[1][1:])



def get_combination(row):
#    feats = [(feat_name, feat_exists) for feat_name, feat_exists in row.items()\
#             if feat_name not in ['Id', 'Response']]
    combin = [0 for _ in range(52)]
    for feat_name, feat_exists in row.items():
        if feat_exists:
            st_num = station_number(feat_name)
            combin[st_num] = 1

    # make into a string
    combin_str = ''
    for val in combin: combin_str += str(val)
    return combin_str



def write_csv(streamobj, data):
    for row in data:
        line = str(row[0])
        for word in row[1:]: line += ',' + str(word)
        streamobj.write(line + '\n')



def sample(rate, date_path, num_path):
    date_buf, num_buf = sio(), sio()
    with open(date_path, 'r') as datefile, open(num_path, 'r') as numfile:

        # write headers
        date_buf.write(datefile.readline())
        num_buf.write(numfile.readline())

        # sample rows
        nlines = 0
        for dateln, numln in izip(datefile, numfile):
            nlines += 1

            if random() < rate:
                date_buf.write(dateln)
                num_buf.write(numln)

            if nlines % 100000 == 0: print("progress: {}".format(nlines))

    date_buf.seek(0)
    num_buf.seek(0)

    return (
        pd.read_csv(date_buf, index_col=0, dtype=np.float32),
        pd.read_csv(num_buf, index_col=0, usecols=list(range(969)), dtype=np.float32)
    )




def sample_uniform(n_rows, cluster_dict, date_path, num_path):
    # determine rates of sampling for each cluster
    row_counts = [tup[1] for tup in cluster_dict.iteritems()]
    row_counts = Counter(row_counts)

    n_clusters = len(row_counts)
    total_rows = len(cluster_dict)

    # rates = [n_rows / (1.*rowc) for rowc in row_counts]
    rates = {cluster: n_rows / (1.*row_counts[cluster]) for cluster in row_counts}

    print "Number of clusters: {}".format(n_clusters)
    print "Total number of rows: {}".format(total_rows)
    print "row counts: {}".format(row_counts)
    print "rates: {}".format(rates)

    # create buffers for each cluster (date_buffer, num_buffer)
    buffers = {cluster: (sio(), sio()) for cluster in rates}

    with open(date_path, 'r') as datefile, open(num_path, 'r') as numfile:

        # write headers
        datehead = datefile.readline()
        numhead = numfile.readline()
        for cluster, (date_buf, num_buf) in buffers.iteritems():
            date_buf.write(datehead)
            num_buf.write(numhead)

        # reading rows
        nlines = 0
        for dateln, numln in izip(datefile, numfile):
            nlines += 1

            rowid = int(numln.split(',', 1)[0])
            cluster = cluster_dict[rowid]

            rate = rates[cluster]
            if random() < rate:
                buffers[cluster][0].write(dateln)
                buffers[cluster][1].write(numln)

            if nlines % 100000 == 0: print("progress: {}".format(nlines))

        for cluster in buffers:
            buffers[cluster][0].seek(0)
            buffers[cluster][1].seek(0)


        dataframes = {}
        for cluster in buffers:
            date_buf, num_buf = buffers[cluster][0], buffers[cluster][1]

            dataframes[cluster] = (
                pd.read_csv(date_buf, index_col=0, dtype=np.float32),
                pd.read_csv(num_buf, index_col=0, usecols=list(range(969)), dtype=np.float32)
            )
            
    return dataframes



def sparse_dummies(categories):
    num_rows = categories.shape[0]
    num_cats = len(categories.values.categories)
    if num_cats == 0: return None
    categories = categories.reset_index(drop=True)[categories.values.codes > -1]
    data = np.ones(categories.shape[0])
    return spar.csr_matrix(
        (data, (categories.index.values, categories.values.codes)),
        shape=(num_rows, num_cats))



def one_hot(df):
    df = df.apply(lambda col: col.astype('category'), axis=0)
    cat_counts = df.apply(lambda col: len(col.values.categories), axis=0)
    header = list(chain.from_iterable(
    [[colname] * ncats for colname, ncats in izip(cat_counts.index, cat_counts.values)]))

    dummy_matrices = (sparse_dummies(df[col]) for col in df.columns)
    # drop None's
    dummy_matrices = list(elem for elem in dummy_matrices if elem is not None)
    
    if len(dummy_matrices) == 0: return None, None
    mtx = spar.hstack(dummy_matrices)
    
    return header, mtx





def save_ohe(filename, header, matrix):
    np.savez(filename,
        header=header,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=matrix.shape)


def load_ohe(filename):
    loader = np.load(filename)
    header = loader['header']
    indices = loader['indices']
    indptr = loader['indptr']
    data = np.ones(len(indices))
    mtx = spar.csr_matrix((data, indices, indptr), shape = loader['shape'])
    return header, mtx





def save_sparse(filename, header, matrix):
    np.savez(filename,
        header=header,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=matrix.shape)


def load_sparse(filename, load_mtx = True):
    loader = np.load(filename)
    header = loader['header']
    if not load_mtx: return header
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    mtx = spar.csc_matrix((data, indices, indptr), shape = loader['shape'])
    return header, mtx





def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))





def to_sparse(df):
    header = df.columns.values

    n_cols = len(header)
    n_rows = len(df.index)

    mtx = df.values  

    rowcrds = []
    colcrds = []

    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(mtx[i,j]):
                rowcrds.append(i)
                colcrds.append(j)

    data = mtx.ravel()
    data = data[~np.isnan(data)]

    row_ind = rowcrds
    col_ind = colcrds

    return header, spar.csc_matrix((data, (row_ind, col_ind)),
        shape = (n_rows, n_cols))




def to_dense_csr(csr_mtx, filler = np.nan):
    dense = np.full(csr_mtx.shape, filler, dtype=np.float64)
    indptr = csr_mtx.indptr
    indices = csr_mtx.indices
    data = csr_mtx.data

    for i in range(len(indptr) - 1):
        col_indices = indices[indptr[i]:indptr[i+1]]
        col_data = data[indptr[i]:indptr[i+1]]
        for j, val in izip(col_indices, col_data):
            dense[i, j] = val

    return dense




def to_dense_csc(csc_mtx, filler = np.nan):
    dense = np.full(csc_mtx.shape, filler, dtype=np.float64)
    indptr = csc_mtx.indptr
    indices = csc_mtx.indices
    data = csc_mtx.data

    for j in range(len(indptr) - 1):
        row_indices = indices[indptr[j]:indptr[j+1]]
        row_data = data[indptr[j]:indptr[j+1]]
        for i, val in izip(row_indices, row_data):
            dense[i, j] = val

    return dense




def get_ifeats(filename):
    ifeats_df = pd.read_csv(
        filename,
        index_col='feat_index',
        usecols=['feat_index', 'final_score']
    )


    ifeats = zip(ifeats_df.index.astype(int).values, ifeats_df['final_score'].values)
    ifeats = sorted(ifeats, key=itemgetter(1), reverse=True)
    feat_indices = list(tup[0] for tup in ifeats)
    feat_importances = list(tup[1] for tup in ifeats)

    return feat_indices, feat_importances






def get_corr_mtx(colnames, mtx, cutoff = 9999):
    # original important features
    important_pairs = [('L3_S32_F3850', 'L3_S33_F3865'), ('L3_S32_F3850', 'L3_S33_F3857'),
                       ('L3_S33_F3859', 'L3_S32_F3850'), ('L3_S33_F3859', 'L3_S29_F3354'),
                       ('L3_S29_F3324', 'L3_S33_F3857'), ('L3_S29_F3376', 'L3_S33_F3865'),
                       ('L3_S29_F3376', 'L3_S33_F3857'), ('L3_S33_F3859', 'L3_S29_F3376'),
                       ('L3_S33_F3859', 'L3_S29_F3324'), ('L3_S29_F3354', 'L3_S33_F3865'),
                       ('L3_S29_F3354', 'L3_S33_F3857'), ('L3_S33_F3859', 'L3_S29_F3321'),
                       ('L3_S29_F3321', 'L3_S33_F3857'), ('L3_S29_F3324', 'L3_S33_F3865'),
                       ('L3_S29_F3321', 'L3_S33_F3865'), ('L3_S30_F3754', 'L3_S33_F3865'),
                       ('L3_S30_F3759', 'L3_S33_F3859'), ('L3_S30_F3759', 'L3_S33_F3865'),
                       ('L3_S30_F3759', 'L3_S33_F3857'), ('L3_S30_F3754', 'L3_S33_F3857'),
                       ('L3_S30_F3754', 'L3_S33_F3859'), ('L3_S33_F3859', 'L3_S35_F3889'),
                       ('L3_S35_F3889', 'L3_S33_F3865'), ('L3_S35_F3889', 'L3_S33_F3857'),
                       ('L2_S26_F3113', 'L3_S33_F3865'), ('L2_S26_F3113', 'L3_S33_F3857'),
                       ('L2_S26_F3113', 'L3_S33_F3859'), ('L3_S32_F3850', 'L3_S38_F3956'),
                       ('L3_S32_F3850', 'L3_S38_F3960'), ('L3_S38_F3952', 'L3_S32_F3850'),
                       ('L0_S0_F22', 'L1_S24_F1846'), ('L0_S0_F22', 'L1_S24_F1844'),
                       ('L1_S24_F1844', 'L0_S1_F28'), ('L0_S1_F28', 'L1_S24_F1846'),
                       ('L0_S5_F114', 'L1_S24_F1846'), ('L0_S5_F114', 'L1_S24_F1844')]


    # # my important features
    # important_pairs = [('L3_S32_F3850', 'L3_S33_F3857'), ('L3_S32_F3850', 'L3_S33_F3865'),
    #                    ('L3_S32_F3850', 'L3_S33_F3859'), ('L3_S29_F3351', 'L3_S33_F3865'),
    #                    ('L3_S29_F3407', 'L3_S33_F3857'), ('L3_S29_F3348', 'L3_S33_F3865'),
    #                    ('L3_S29_F3348', 'L3_S33_F3859'), ('L3_S29_F3339', 'L3_S33_F3865'),
    #                    ('L3_S29_F3351', 'L3_S33_F3857'), ('L3_S29_F3339', 'L3_S33_F3857'),
    #                    ('L3_S29_F3339', 'L3_S33_F3859'), ('L3_S29_F3407', 'L3_S33_F3865'),
    #                    ('L3_S29_F3348', 'L3_S33_F3857'), ('L3_S29_F3351', 'L3_S33_F3859'),
    #                    ('L3_S29_F3407', 'L3_S33_F3859'), ('L3_S30_F3829', 'L3_S33_F3859'),
    #                    ('L3_S30_F3809', 'L3_S33_F3859'), ('L3_S30_F3829', 'L3_S33_F3865'),
    #                    ('L3_S30_F3829', 'L3_S33_F3857'), ('L3_S30_F3769', 'L3_S33_F3857'),
    #                    ('L3_S30_F3769', 'L3_S33_F3865'), ('L3_S30_F3769', 'L3_S33_F3859'),
    #                    ('L3_S30_F3809', 'L3_S33_F3857'), ('L3_S30_F3809', 'L3_S33_F3865'),
    #                    ('L3_S30_F3754', 'L3_S33_F3857'), ('L3_S30_F3754', 'L3_S33_F3859'),
    #                    ('L3_S30_F3744', 'L3_S33_F3865'), ('L3_S30_F3744', 'L3_S33_F3857'),
    #                    ('L3_S30_F3754', 'L3_S33_F3865'), ('L3_S30_F3744', 'L3_S33_F3859'),
    #                    ('L3_S30_F3759', 'L3_S33_F3865'), ('L3_S30_F3759', 'L3_S33_F3859'),
    #                    ('L3_S30_F3759', 'L3_S33_F3857'), ('L3_S30_F3704', 'L3_S33_F3859'),
    #                    ('L3_S30_F3704', 'L3_S33_F3865'), ('L3_S30_F3704', 'L3_S33_F3857')]

    pairs = important_pairs

    pair_idx = list(
        ( np.where(colnames==pair[0])[0][0] , np.where(colnames==pair[1])[0][0] ) for pair in pairs
    )

    pair_idx = pair_idx[:cutoff]

    relevant_cols = list(set(chain.from_iterable(pair_idx)))

    sub_mtx = mtx[:, relevant_cols]

    sub_mtx = to_dense_csc(sub_mtx)

    exists = np.int32(~np.isnan(sub_mtx)) * 2 - 1


    corr_cols = []
    for left, right in pair_idx:
        left  = np.where(relevant_cols== left)[0][0]
        right = np.where(relevant_cols==right)[0][0]

        corr_col = exists[:, left] * exists[:, right]

        corr_cols.append(corr_col[:, np.newaxis])

    return np.hstack(corr_cols)




def sample_mtx_rows(mtx, fraction):
    total = mtx.shape[0]
    sub_size = int(fraction*total)
    idx = np.arange(total)
    sub_idx = np.random.choice(idx, size=sub_size, replace=False)
    return mtx[sub_idx]






