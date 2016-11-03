import numpy as np
from argparse import ArgumentParser
from tools import load_sparse, get_ifeats


parser = ArgumentParser()

parser.add_argument('num', metavar='NUM', type=int)
args = parser.parse_args()

num = args.num

colnames = load_sparse('../input/train_numeric_sparse', False)
ifeats, importances = get_ifeats('num_feature_importances.csv')
ifeats = ifeats[:num]


print list(colnames[idx] for idx in ifeats)


