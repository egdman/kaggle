from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser()

parser.add_argument('more', metavar="MORE_IMPORTANT", type=str)
parser.add_argument('less', metavar="LESS_IMPORTANT", type=str)

args = parser.parse_args()

more_path = args.more
less_path = args.less

less_df = pd.read_csv(less_path, index_col='Id')
more_df = pd.read_csv(more_path, index_col='Id')

# make all zeros
more_df['Response'] = [0 for _ in range(more_df.shape[0])]

print "merging {} {} into {} {}...".format(more_path, more_df.shape, less_path, less_df.shape)

less_df.loc[more_df.index, 'Response'] = more_df['Response']

less_df.to_csv("merged_predictions.csv")