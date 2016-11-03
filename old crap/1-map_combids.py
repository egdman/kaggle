import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from itertools import izip
from tools import *

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-test', action='store_true')

args = parser.parse_args()
test = args.test

word = 'test' if test else 'train'

print("processing %s data..." % word)

num_path = '../input/%s_numeric.csv' % word
date_path = '../input/%s_date.csv' % word


NROWS = 2000000 # total number of rows to read
CHUNK = 10000
uniq_combinations = set()


all_ids = []
all_combinations = []


lines_read = 0
for date_df, num_df in izip(
	pd.read_csv(date_path, dtype=np.float32, chunksize=CHUNK),
	pd.read_csv(num_path, dtype=np.float32, chunksize=CHUNK)
	):

    all_ids.extend(date_df['Id'].astype(np.int64).values)

    date_df = date_df.notnull().drop(['Id'], axis=1)

    for daterow in date_df.iterrows():    
        daterow = daterow[1].to_dict()
        combination = get_combination(daterow)
        uniq_combinations.add(combination)
        all_combinations.append(combination)
        
    lines_read += CHUNK
    
    if lines_read % 50000 == 0: print("progress: {0}".format(lines_read))
    if lines_read >= NROWS: break



uniq_combinations = list(enumerate(sorted(uniq_combinations)))
print("found {0} combinations".format(len(uniq_combinations)))


id_comb_res_path = "id-comb-%s.csv" % word
enum_comb_path = "enumerated_combinations_%s.csv" % word

with open(enum_comb_path, 'w') as resfile:
    resfile.write("combid,combination\n")
    write_csv(resfile, uniq_combinations)
    
with open(id_comb_res_path, 'w') as resfile:
    resfile.write("Id,combination\n")
    write_csv(resfile, zip(all_ids, all_combinations))