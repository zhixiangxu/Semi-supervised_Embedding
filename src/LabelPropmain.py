from sklearn import datasets
from sklearn.semi_supervised import label_propagation
from sklearn import preprocessing

import numpy as np
import random
import scipy.io as sio


def load_paviaU(dconf):
    shape_ = dconf['shape']
    ifscale = dconf['scale']

    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/PaviaU.mat'
    dset = sio.loadmat(fn)
    if ifscale == False:
        x = dset['paviaU']
    else:
        x = dset['paviaU'].reshape(shape_[0]*shape_[1], shape_[2])
        x = np.array(x, dtype=np.float)
        x = preprocessing.scale(x)
        x = x.reshape(shape_)
    x = x.reshape(shape_[0]*shape_[1], shape_[2])

    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/PaviaU_gt.mat'
    dset = sio.loadmat(fn)
    y = dset['paviaU_gt']
    y = y.flatten()

    idxLabs, idxUnls = [], []
    for i in range(shape_[0] * shape_[1]):
        if int(y[i]) == 0:
            idxUnls.append(i)
        else:
            idxLabs.append(i)

    # select labeled and unlabeled
    train, test = {}, {}
    a = list(np.arange(len(idxLabs)))
    b = random.sample(a, int(0.8*len(a)))
    c = list(set(a).difference(set(b)))

    # traing data
    train['x'] = np.array([x[idxLabs[i]] for i in b] + [x[i] for i in idxUnls])
    train['y'] = np.array([y[idxLabs[i]] for i in b] + [y[i] for i in idxUnls], dtype=int)
    train['y'] -= 1

    shf = np.arange(len(train['y']))
    np.random.shuffle(shf)
    train['x'] = train['x'][shf]
    train['y'] = train['y'][shf]

    maxnum = 10000
    train['x'] = train['x'][:maxnum]
    train['y'] = train['y'][:maxnum]

    # test data
    test['x'] = np.array([x[idxLabs[i]] for i in c])
    test['y'] = np.array([y[idxLabs[i]] for i in c], dtype=int)
    test['y'] -= 1

    return train, test

def load_indian(dconf):
    shape_ = dconf['shape']
    ifscale = dconf['scale']

    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/Indian_pines_corrected.mat'
    dset = sio.loadmat(fn)
    if ifscale == False:
        x = dset['indian_pines_corrected']
    else:
        x = dset['indian_pines_corrected'].reshape(shape_[0]*shape_[1], shape_[2])
        x = np.array(x, dtype=np.float)
        x = preprocessing.scale(x)
        x = x.reshape(shape_)
    x = x.reshape(shape_[0]*shape_[1], shape_[2])

    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/Indian_pines_gt.mat'
    dset = sio.loadmat(fn)
    y = dset['indian_pines_gt']
    y = y.flatten()

    idxLabs, idxUnls = [], []
    for i in range(shape_[0] * shape_[1]):
        if int(y[i]) == 0:
            idxUnls.append(i)
        else:
            idxLabs.append(i)

    # select labeled and unlabeled
    train, test = {}, {}
    a = list(np.arange(len(idxLabs)))
    b = random.sample(a, int(0.8*len(a)))
    c = list(set(a).difference(set(b)))

    # traing data
    train['x'] = np.array([x[idxLabs[i]] for i in b] + [x[i] for i in idxUnls])
    train['y'] = np.array([y[idxLabs[i]] for i in b] + [y[i] for i in idxUnls], dtype=int)
    train['y'] -= 1

    shf = np.arange(len(train['y']))
    np.random.shuffle(shf)
    train['x'] = train['x'][shf]
    train['y'] = train['y'][shf]

    maxnum = 10000
    train['x'] = train['x'][:maxnum]
    train['y'] = train['y'][:maxnum]

    # test data
    test['x'] = np.array([x[idxLabs[i]] for i in c])
    test['y'] = np.array([y[idxLabs[i]] for i in c], dtype=int)
    test['y'] -= 1

    return train, test

from PIL import Image
def load_Houston(dconf):
    shape_ = dconf['shape']
    ifscale = dconf['scale']

    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/CASI.mat'
    dset = sio.loadmat(fn)
    if ifscale == False:
        x = dset['CASI_SC']
    else:
        x = dset['CASI_SC'].reshape(shape_[0]*shape_[1], shape_[2])
        x = np.array(x, dtype=np.float)
        x = preprocessing.scale(x)
        x = x.reshape(shape_)
    x = x.reshape(shape_[0]*shape_[1], shape_[2])

    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/Train_Houston.tif'
    im = Image.open(fn)
    y1 = np.array(im).astype(np.int32)-1
    y1 = y1.flatten()

    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/Test_Houston.tif'
    im = Image.open(fn)
    y2 = np.array(im).astype(np.int32)-1
    y2 = y2.flatten()

    idxTrai, idxUnls, idxTest = [], [], []
    for i in range(shape_[0]*shape_[1]):
        if int(y1[i]) == -1 and int(y2[i]) == -1:
            idxUnls.append(i)
        elif int(y1[i]) != -1:
            idxTrai.append(i)
        elif int(y2[i]) != -1:
            idxTest.append(i)

    # select labeled and unlabeled
    train, test = {}, {}

    # traing data
    train['x'] = np.array([x[i] for i in idxTrai] + [x[i] for i in idxUnls])
    train['y'] = np.array([y1[i] for i in idxTrai] + [-1 for i in idxUnls], dtype=np.int64)

    shf = np.arange(len(train['y']))
    np.random.shuffle(shf)
    train['x'] = train['x'][shf]
    train['y'] = train['y'][shf]

    maxnum = 5000
    train['x'] = train['x'][:maxnum]
    train['y'] = train['y'][:maxnum]

    # test data
    test['x'] = np.array([x[i] for i in idxTest])
    test['y'] = np.array([y2[i] for i in idxTest], dtype=np.int64)

    return train, test


import json
with open('./config/houston.json', 'r') as f:
    dconf = json.load(f)

def load_data(dconf):
    name = dconf['name']
    fdict = {
        'paviaU': load_paviaU,
        'indian': load_indian,
        'houston': load_Houston
    }
    return fdict[name](dconf)

from utils import int_accuracy

def main():
    print('loading dataset')
    train, test = load_data(dconf)

    print('training model')
    lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=5)
    lp_model.fit(train['x'], train['y'])

    print('testing model')
    pred = lp_model.predict(test['x'])

    print(int_accuracy(test['y'], pred))

    for i in range(len(pred)):
        print((pred[i], test['y'][i]), end='')


if __name__ == '__main__':
    main()
