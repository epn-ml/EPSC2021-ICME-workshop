
import pandas as pds
import numpy as np
import time
import datetime
import event as evt
from scipy.signal import find_peaks, peak_widths
from joblib import Parallel, delayed
from lmfit import models
import numpy.random as random
from tqdm import tqdm
from data_generator import *
import pickle

def get_catevents():
    
    # load ICME catalog data
    [ic,header,parameters] = pickle.load(open('data/HELCATS_ICMECAT_v20_pandas.p', "rb" ))
    # extract important values
    isc = ic.loc[:,'sc_insitu'] 
    starttime = ic.loc[:,'icme_start_time']
    endtime = ic.loc[:,'mo_end_time']
    # Event indices
    iwinind = np.where(isc == 'Wind')[0]
    istaind = np.where(isc == 'STEREO-A')[0]
    istbind = np.where(isc == 'STEREO-B')[0]

    winbegin = starttime[iwinind]
    winend = endtime[iwinind]

    stabegin = starttime[istaind]
    staend = endtime[istaind]

    stbbegin = starttime[istbind]
    stbend = endtime[istbind]

    # get list of events

    evtListw = evt.read_cat(winbegin, winend, iwinind)
    evtLista = evt.read_cat(stabegin, staend, istaind)
    evtListb = evt.read_cat(stbbegin, stbend, istbind)
    
    return evtListw, evtLista, evtListb    
    
def generate_result(test_image_paths, test_mask_paths, model):
    ## Generating the result
    image_size = (1024,1,10)
    for i, path in tqdm(enumerate(test_image_paths), total=len(test_image_paths)):
    
        df_mask = pds.read_csv(test_mask_paths[i],header = None,index_col=[0])
        image = parse_image(test_image_paths[i], image_size)
        predict_mask = model.predict(np.expand_dims(image, axis=0))[0]
        df_mask['pred'] = np.squeeze(predict_mask)
        df_mask.columns = ['true', 'pred']
        if i == 0:
            result = df_mask
        else:
            result = pds.concat([result, df_mask], sort=True)

    result = result.sort_index()
    result.index = pds.to_datetime(result.index)
    
    return result

def removeCreepy(eventList, thres=2):
    '''
    For a given list, remove the element whose duration is under the threshold
    '''
    return [x for x in eventList if x.duration > datetime.timedelta(hours=thres)]

def make_binary(serie, thresh):
    
    serie = (serie > thresh)*1
    serie = serie.interpolate()
    
    return serie

def makeEventList(y, label, delta=2):
    '''
    Consider y as a pandas series, returns a list of Events corresponding to
    the requested label (int), works for both smoothed and expected series
    Delta corresponds to the series frequency (in our basic case with random
    index, we consider this value to be equal to 2)
    '''
    listOfPosLabel = y[y == label]
    if len(listOfPosLabel) == 0:
        return []
    deltaBetweenPosLabel = listOfPosLabel.index[1:] - listOfPosLabel.index[:-1]
    deltaBetweenPosLabel.insert(0, datetime.timedelta(0))
    endOfEvents = np.where(deltaBetweenPosLabel > datetime.timedelta(minutes=delta))[0]
    indexBegin = 0
    eventList = []
    for i in endOfEvents:
        end = i
        eventList.append(evt.Event(listOfPosLabel.index[indexBegin], listOfPosLabel.index[end]))
        indexBegin = i+1
    eventList.append(evt.Event(listOfPosLabel.index[indexBegin], listOfPosLabel.index[-1]))
    return eventList

def get_truelabel(data,events):
    
    x = pds.to_datetime(data.index)
    y = np.zeros(np.shape(data)[0])
    
    for e in events:
        n_true = np.where((x >= e.begin) & (x <= e.end))
        y[n_true] = 1
    
    label = pds.DataFrame(y, index = data.index, columns = ['label'])
    
    return label