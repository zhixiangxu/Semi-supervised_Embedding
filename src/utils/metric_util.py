import numpy as np
import keras.metrics as KM

def int_accuracy(yl, yg):
    cnt = 0
    length = len(yg)
    for i in range(length):
        if int(yg[i]) == int(yl[i]):
            cnt += 1
    return 'accuracy', cnt/length

def int_precision(yl, yg):
    pass

def int_recall(yl, yg):
    pass

def int_fbeta_score(yl, yg):
    pass

def mean_absolute_error(yl, yg):
    return KM.mean_absolute_error(yl, yg)