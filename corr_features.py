from tools import *

import gc
import numpy as np
import pandas as pd
import ast
from itertools import chain



header, mtx = load_sparse('../input/train_numeric_sparse', True)
print mtx.shape

# sub_mtx = mtx[:20]
# del mtx
# gc.collect()

if header[-1] == 'Response' : header = header[:-1]

corr_mtx = get_corr_mtx(header, mtx)

print corr_mtx.shape








