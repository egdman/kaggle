import pandas as pd
import scipy.sparse as spar
import numpy as np
from itertools import izip, chain
import pickle
import re
from tools import to_sparse, save_sparse, load_sparse, chunker
from os import path, mkdir, listdir
from operator import itemgetter
from time import sleep
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('input_path', metavar='FILE', type=str)
args = parser.parse_args()


data_path = args.input_path

in_dirpath = path.dirname(data_path)
in_filename = path.basename(data_path)

temp_dirname = in_filename + '_files'
temp_dirpath = path.join(in_dirpath, temp_dirname)

print "temp directory: {}".format(temp_dirpath)


# if the directory does not exist yet, create it and write files
try:
	mkdir(temp_dirpath)

	chunksize = 30

	columns = pd.read_csv(
		data_path,
		index_col=0,
		nrows = 1,
		dtype=np.float32).columns.values

	ncols = len(columns)
	print(ncols)


	headers = []
	matrices = []
	progress = 0
	counter = 0
	# read file in vertical chunks
	for col_subset in chunker(columns, chunksize):

		progress += chunksize
		counter += 1

		usecols = np.append(['Id'], col_subset)
		df = pd.read_csv(
			data_path,
			index_col=0,
			dtype=np.float32,
			# nrows=100000,
			usecols=usecols
		)

		header, mtx = to_sparse(df)
		if header is not None:
			matrices.append(mtx)
			headers.append(header)
			print("progress = {:.1f}% : {}".format(100.*progress / (1.*ncols), mtx.shape))
			print mtx.__class__
			fname = str(counter) + '_temp'
			save_sparse(path.join(temp_dirpath, fname), header, mtx)



	#######################################################################

# if the temp directory already exists, use it
except OSError:
	pass


fnames = listdir(temp_dirpath)

fnums = (int(match.group(0)) if match is not None else -1 \
		for match in (re.search('\d+', fname) for fname in fnames))

fnames = list(path.join(temp_dirpath, tup[1]) for tup in \
	sorted(izip(fnums, fnames), key = itemgetter(0)) \
	if tup[0] >= 0)


headers = []
matrices = []
for fname in fnames:
	print fname
	header, mtx = load_sparse(fname)
	headers.append(header)
	matrices.append(mtx)
	print mtx.__class__


print "loading complete"
sleep(20)
print "hstack started"

full_header = np.hstack(headers)
full_mtx = spar.hstack(matrices)

print full_header.shape
print full_mtx.shape
print full_mtx.__class__


out_path = re.search(r'(.*)(?=\.(.*))', data_path).group(1) + '_sparse'
print "writing output to file {}".format(out_path)

with open(out_path, 'wb') as resfile:
	save_sparse(resfile, full_header, full_mtx)