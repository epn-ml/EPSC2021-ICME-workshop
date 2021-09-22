import pandas as pds
import datetime
import numpy as np
import time
import event as evt
import numpy as np
import math
from numpy.lib.stride_tricks import as_strided


'''
Utilities used for transforming a dataset in sliding window in a smart way,
less memory costing than what was done for the sliding windows
'''


def windowed(X, window):
    '''
    Using stride tricks to create a windowed view on the original
    data.
    '''
    shape = int((X.shape[0] - window) + 1), window, X.shape[1]
    strides = (X.strides[0],) + X.strides
    X_windowed = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    return X_windowed

def windowedU(X, window):
    n = len(X)/window
    if n == math.trunc(n):
        a = np.array_split(X, n)
        X_windowed = [x for x in a]    
    else:
        a = np.array_split(X, np.ceil(n))
        X_windowed = [x for x in a if len(x) == window]
        for x in a:
            print(len(x))
    return X_windowed

def windowedUnet(X,window):
    X_windowed = [X[x:x+window] for x in range(0, len(X)-window, window)]
    return np.asarray(X_windowed)

def make_views(arr,win_size,step_size,writeable = False):
    """
    arr: any 2D array whose columns are distinct variables and 
    rows are data records at some timestamp t
    win_size: size of data window (given in data points along record/time axis)
    step_size: size of window step (given in data point along record/time axis)
    writable: if True, elements can be modified in new data structure, which will affect
    original array (defaults to False)
  
    Note that step_size is related to window overlap (overlap = win_size - step_size), in 
    case you think in overlaps.
  
    This function can work with C-like and F-like arrays, and with DataFrames.  Yay.
    """
  
    # If DataFrame, use only underlying NumPy array
    if type(arr) == type(pds.DataFrame()):
        arr['index'] = arr.index
        arr = arr.values
  
    # Compute Shape Parameter for as_strided
    n_records = arr.shape[0]
    n_columns = arr.shape[1]
    remainder = (n_records - win_size) % step_size 
    num_windows = 1 + int((n_records - win_size - remainder) / step_size)
    shape = (num_windows, win_size, n_columns)
  
    # Compute Strides Parameter for as_strided
    next_win = step_size * arr.strides[0]
    next_row, next_col = arr.strides
    strides = (next_win, next_row, next_col)

    new_view_structure = as_strided(arr,shape = shape,strides = strides,writeable = writeable)
    return new_view_structure