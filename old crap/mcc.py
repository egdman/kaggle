from argparse import ArgumentParser
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix

parser = ArgumentParser()

parser.add_argument('pred', metavar='PRED', type=str)
parser.add_argument('true', metavar='TRUE', type=str)

args = parser.parse_args()

pred = pd.read_csv(args.pred, index_col='Id', usecols=['Id', 'Response'], dtype=int)['Response']
true = pd.read_csv(args.true, index_col='Id', usecols=['Id', 'Response'], dtype=int)['Response']

true = true.loc[pred.index]

print "pred: {}".format(pred.shape)
print "true: {}".format(true.shape)

print "\nMCC = {}".format(matthews_corrcoef(true.values.ravel(), pred.values.ravel()))

conmtx = confusion_matrix(true.values.ravel(), pred.values.ravel())

print "\nconfusion mtx:"
print "TN: {:>7} FP: {:>7}".format(conmtx[0,0], conmtx[0,1])
print "FN: {:>7} TP: {:>7}".format(conmtx[1,0], conmtx[1,1])
print
print "True  positive rate: {:.5f}".format(conmtx[1,1] / float(conmtx[1,1] + conmtx[1,0]))
print "False positive rate: {:.5f}".format(conmtx[0,1] / float(conmtx[0,1] + conmtx[0,0]))