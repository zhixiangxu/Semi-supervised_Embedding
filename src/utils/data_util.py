import scipy.io as sio
import random
import numpy as np
from sklearn import preprocessing
from sklearn import datasets as skdataset
import utils

def Euclidean_distance(x, y):
    return np.linalg.norm(x-y)

dist = {'euclidean': Euclidean_distance, }

def dataset(dconf):
    name = dconf['name']
    if name in ('g50c','mnist'):
        return Basic(dconf)
    elif name in ('paviaU', 'indian', 'houston'):
        return Remote(dconf)
    else:
        print('ERROR: wrong dataset name')
        return None

class Scikit(object):
    def __init__(self, dconf):
        self.name = dconf['name']
        fdict = {'iris': skdataset.load_iris, }
        self.train, self.test = fdict[self.name]()

    def OneHot(self, data, width):
        tmp = np.zeros((data.size, width))
        tmp[np.arange(data.size), data] = 1
        return tmp

def load_iris():
    rng = np.random.RandomState(0)
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    y_30 = np.copy(y)
    y_30[rng.rand(len(y)) < 0.3] = -1

    idxTrai, idxTest, idxUnls = [], [], []

    for i in range(len(y_30)):
        if y_30[i] == -1:
            idxUnls.append(i)
        else:
            idxTrai.append(i)
    
    tmp = np.arange(len(idxTrai))
    np.random.shuffle(tmp)
    cutpt = int(0.8 * len(tmp))

    train, test = {}, {}
    train['x'] = np.array(list(X[idxTrai[tmp[:cutpt]]]) + list(x[idxUnls]))
    train['y'] = None

    test['x'] = None

class Basic(object):
    def __init__(self, dconf):
        self.name = dconf['name']
        fdict = {'g50c': load_g50c, 'mnist':load_mnist,}
        self.train, self.test = fdict[self.name]()
        # for training
        # global set in class:
        # 0 - labeled, 1 - unlabeled
        self.length = [len(self.train[key]) for key in ('xl', 'xu')]
        self.knn = 10
        self.keydict = {0:'xl', 1:'xu'}
        self.ptr = 0

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def next(self):        
        labeled, unlabeled = {}, {}
        labeled['x'] = self.train['xl'][: self.batch_size[0]]
        labeled['y'] = self.train['y'][: self.batch_size[0]]
        unlabeled['x'] = self.train['xu'][: self.batch_size[1]]

        np.roll(self.train['xl'], -self.batch_size[0])
        np.roll(self.train['y'], -self.batch_size[0])
        np.roll(self.train['xu'], -self.batch_size[1])

        return labeled, unlabeled

    def find_neighbor(self, p):
        # pick a random pair of neighbors of $x_i$, $x_j$
        # from k-nearest neighbors
        distlst = []
        for i in [0, 1]:
            key = self.keydict[i]
            distlst += [dist['euclidean'](self.train[key][p], self.train[key][j]) for j in range(self.length[i])]
        distlst = np.array(distlst)
        knnlst = distlst.argsort()[:self.knn]
        q = np.random.randint(0, self.knn-1)
        q = knnlst[q]
        if q >= self.length[0]:
            return self.train['xu'][q - self.length[0]]
        else:
            return self.train['xl'][q]

    def pick(self, pick_reshape):
        x_reshape = pick_reshape['x']
        y_reshape = pick_reshape['y']

        # pick a labeled example
        i = self.ptr%self.length[0]
        xl = self.train['xl'][i].reshape(x_reshape)
        yt = self.train['y'][i].reshape(y_reshape)
        # pick a random neighbor
        xn = self.find_neighbor(i).reshape(x_reshape)
        # pick a random unlabeled`
        i = np.random.randint(0, self.length[0]-1)
        xu = self.train['xu'][i].reshape(x_reshape)
        # update pointer
        self.ptr += 1
        return xl, xn, xu, yt

def load_mnist():    
    train_x = utils.mnistdata(filename='/home/yangminz/Semi-supervised_Embedding/dataset/MNIST/train-images.idx3-ubyte').getImage()
    train_y = utils.mnistdata(filename='/home/yangminz/Semi-supervised_Embedding/dataset/MNIST/train-labels.idx1-ubyte').getLabel()
    test_x = utils.mnistdata('/home/yangminz/Semi-supervised_Embedding/dataset/MNIST/t10k-images.idx3-ubyte').getImage()
    test_y = utils.mnistdata('/home/yangminz/Semi-supervised_Embedding/dataset/MNIST/t10k-labels.idx1-ubyte').getLabel()

    def OneHot(flag, label):
        tmp = None
        if flag == True:
            tmp = np.zeros((len(label), 10))
            tmp[np.arange(len(label)), label] = 1
            tmp = np.array(tmp, dtype=np.int)
        else:
            tmp = label
        return tmp
    
    train, test = {}, {}
    k = int(0.1 * len(train_x))
    # traing data
    train['xl'] = train_x[:k].reshape(-1, 28, 28, 1)
    train['y'] = OneHot(True, train_y[:k])
    train['xu'] = train_x[k:].reshape(-1, 28, 28, 1)
    # test data
    test['x'] = test_x.reshape(-1, 28, 28, 1)
    test['y'] = OneHot(False, test_y)

    return train, test

def load_g50c():
    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/g50c.mat'
    dset = sio.loadmat(fn)
    '''
    read data from matlab file:
    - X = matrix of input data; each row corresponds to one example
    - y = the labels
    - idxLabs = each row contains the indices of the labeled points for a given split
    - idxUnls = idem for the unlabeled points
    '''
    X = dset['X']
    X = preprocessing.scale(X)
    y = dset['y']
    def if_make_zero(flag):
        if flag == True:
            for i in range(len(y)):
                if int(y[i]) == -1:
                    y[i] = 0
    if_make_zero(False)
    # matlab take index from 1
    idxLabs = dset['idxLabs'] - 1
    idxUnls = dset['idxUnls'] - 1
    # select labeled and unlabeled
    train, test = {}, {}
    lenl, lenu = len(idxLabs[0]), len(idxUnls[0])
    fold = 0.8
    # traing data
    train['xl'] = np.array([X[i] for i in idxLabs[0]])
    train['y'] = np.array([y[i] for i in idxLabs[0]])
    train['xu'] = np.array([X[i] for i in idxUnls[0][:int(fold*lenu)]])
    # test data
    test['x'] = np.array([X[i] for i in idxUnls[0][int(fold*lenu):]])
    test['y'] = np.array([y[i] for i in idxUnls[0][int(fold*lenu):]])

    return train, test

class Remote(object):
    def __init__(self, dconf):
        self.name = dconf['name']
        self.pick_reshape = dconf['pick_reshape']

        fdict = {'paviaU': load_paviaU, 'indian':load_indian, 'houston':load_Houston2}
        self.x, self.y, self.train, self.test = fdict[self.name](dconf['scale'])
        # for training
        # global set in class:
        # 0 - labeled, 1 - unlabeled
        self.length = [len(self.train[key]) for key in ('xl', 'xu')]
        self.keydict = {0:'xl', 1:'xu'}
        self.ptr = 0

    def OneHot(self, data, width):
        tmp = np.zeros((data.size, width))
        tmp[np.arange(data.size), data] = 1
        return tmp

    def find_neighbor(self, i, j):
        ni = np.random.random_integers(i-self.radius, i+self.radius)
        nj = np.random.random_integers(j-self.radius, j+self.radius)
        flag = (ni==i and nj==j) or (ni<0 or nj <0) or (ni >= self.x.shape[0] or nj >= self.x.shape[1])
        while flag:
            ni = np.random.random_integers(i-self.radius, i+self.radius)
            nj = np.random.random_integers(j-self.radius, j+self.radius)
            #print((i,j),(ni,nj))
            flag = (ni == i and nj == j) or (ni < 0 or nj < 0) or (ni >= self.x.shape[0] or nj >= self.x.shape[1])
        return self.x[ni][nj]

    def pick(self):
        x_reshape = self.pick_reshape['x']
        y_reshape = self.pick_reshape['y']

        # pick a labeled example
        self.radius = 1
        i, j = self.train['xl'][np.random.randint(0, self.length[0]-1)]
        xl = self.x[i][j].reshape(x_reshape)
        yt = self.y[i][j]
        yt = self.OneHot(yt, y_reshape[1]).reshape(y_reshape)
        # pick a random neighbor
        xn = self.find_neighbor(i, j).reshape(x_reshape)
        # pick a random unlabeled
        s = np.random.randint(0, self.length[1]-1)
        i, j = self.train['xu'][s]
        xu = self.x[i][j].reshape(x_reshape)
        # update pointer
        self.ptr += 1
        return xl, xn, xu, yt

    def next_batch(self, nb_batch):
        x_reshape = self.pick_reshape['x']
        y_reshape = self.pick_reshape['y']

        x = np.array([self.x[i][j] for i,j in self.train['xl'][: nb_batch]])
        y = np.array([self.y[i][j] for i,j in self.train['xl'][: nb_batch]])
        y = self.OneHot(y, y_reshape[1])
        xu = np.array([self.x[i][j] for i,j in self.train['xu'][: nb_batch]])

        np.roll(self.train['xl'], -nb_batch)
        np.roll(self.train['xu'], -nb_batch)

        tmp = np.vstack([x, xu])

        return tmp, y

    def get_test(self, do_one_hot):
        x = self.test['x']
        y = self.test['y']
        if do_one_hot:
            y = self.OneHot(y, self.pick_reshape['y'][1])
        return x, y

def load_paviaU(ifscale):
    shape_ = [610, 340, 103]

    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/PaviaU.mat'
    dset = sio.loadmat(fn)
    if ifscale == False:
        x = dset['paviaU']
    else:
        x = dset['paviaU'].reshape(shape_[0]*shape_[1], shape_[2])
        x = preprocessing.scale(x)
        x = x.reshape(shape_)
    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/PaviaU_gt.mat'
    dset = sio.loadmat(fn)
    y = dset['paviaU_gt'].astype(np.int32) - 1

    idxLabs, idxUnls = [], []
    #x_ = np.zeros([shape_[0], shape_[1], 50])
    for i in range(shape_[0]):
        for j in range(shape_[1]):
            # tmp = x[i][j][subsample]
            # x_[i][j] = tmp
            if int(y[i][j]) == -1:
                idxUnls.append([i,j])
            else:
                idxLabs.append([i,j])

    # select labeled and unlabeled
    train, test = {}, {}
    a = list(np.arange(len(idxLabs)))
    b = random.sample(a, int(0.8*len(a)))
    c = list(set(a).difference(set(b)))

    # traing data index
    train['xl'] = [idxLabs[i] for i in b]
    train['xu'] = idxUnls
    # test data
    test['x'] = [x[idxLabs[i][0]][idxLabs[i][1]] for i in c]
    test['y'] = np.array([y[idxLabs[i][0]][idxLabs[i][1]] for i in c], dtype=np.int32)

    return x, y, train, test

def load_indian(ifscale):
    shape_ = [145, 145, 200]

    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/Indian_pines_corrected.mat'
    dset = sio.loadmat(fn)
    if ifscale == False:
        x = dset['indian_pines_corrected']
    else:
        x = dset['indian_pines_corrected'].reshape(shape_[0]*shape_[1], shape_[2])
        x = preprocessing.scale(x)
        x = x.reshape(shape_)
    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/Indian_pines_gt.mat'
    dset = sio.loadmat(fn)
    y = dset['indian_pines_gt'].astype(np.int32) - 1

    idxLabs, idxUnls = [], []
    for i in range(shape_[0]):
        for j in range(shape_[1]):
            if int(y[i][j]) == -1:
                idxUnls.append([i,j])
            else:
                idxLabs.append([i,j])

    # select labeled and unlabeled
    train, test = {}, {}
    a = list(np.arange(len(idxLabs)))
    b = random.sample(a, int(0.8*len(a)))
    c = list(set(a).difference(set(b)))

    # traing data index
    train['xl'] = [idxLabs[i] for i in b]
    train['xu'] = idxUnls
    # test data
    test['x'] = [x[idxLabs[i][0]][idxLabs[i][1]] for i in c]
    test['y'] = np.array([y[idxLabs[i][0]][idxLabs[i][1]] for i in c], dtype=np.int32)

    return x, y, train, test

from PIL import Image
def load_Houston(ifscale):
    shape_ = [349, 1905, 144]

    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/CASI.mat'
    dset = sio.loadmat(fn)
    if ifscale == False:
        x = dset['CASI_SC']
    else:
        x = dset['CASI_SC'].reshape(shape_[0]*shape_[1], shape_[2])
        x = preprocessing.scale(x)
        x = x.reshape(shape_)

    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/Train_Houston.tif'
    im = Image.open(fn)
    y1 = np.array(im).astype(np.int32)-1

    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/Test_Houston.tif'
    im = Image.open(fn)
    y2 = np.array(im).astype(np.int32)-1

    idxTrai, idxUnls, idxTest = [], [], []
    for i in range(shape_[0]):
        for j in range(shape_[1]):
            if int(y1[i][j]) == -1 and int(y2[i][j]) == -1:
                idxUnls.append([i,j])
            elif int(y1[i][j]) != -1:
                idxTrai.append([i,j])
            elif int(y2[i][j]) != -1:
                idxTest.append([i,j])

    # select labeled and unlabeled
    train, test = {}, {}

    # traing data index
    train['xl'] = idxTrai
    train['xu'] = idxUnls
    # test data
    test['x'] = np.array([x[i][j] for i,j in idxTest])
    test['y'] = np.array([y1[i][j] for i,j in idxTest], dtype=np.int32)

    return x, y1, train, test

def load_Houston2(ifscale):
    shape_ = [349, 1905, 144]

    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/CASI.mat'
    dset = sio.loadmat(fn)
    if ifscale == False:
        x = dset['CASI_SC']
    else:
        x = dset['CASI_SC'].reshape(shape_[0]*shape_[1], shape_[2])
        x = preprocessing.scale(x)
        x = x.reshape(shape_)

    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/Train_Houston.tif'
    im = Image.open(fn)
    y1 = np.array(im).astype(np.int32)

    fn = '/home/yangminz/Semi-supervised_Embedding/dataset/Test_Houston.tif'
    im = Image.open(fn)
    y2 = np.array(im).astype(np.int32)

    y = y1 + y2 - 1

    idxLabs, idxUnls, idxTest = [], [], []
    idxClas = {}
    for i in range(shape_[0]):
        for j in range(shape_[1]):
            if int(y[i][j]) == -1:
                idxUnls += [[i,j]]
                continue
            for c in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
                if int(y[i][j]) == c:
                    if c not in idxClas.keys():
                        idxClas[c] = [[i,j]]
                    else:
                        idxClas[c] += [[i,j]]
                        continue

    for c in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        np.random.shuffle(idxClas[c])
        cut_pt = int(0.8*len(idxClas[c]))
        idxLabs += idxClas[c][: cut_pt]
        idxTest += idxClas[c][cut_pt: ]
    np.random.shuffle(idxLabs)
    np.random.shuffle(idxTest)
    np.random.shuffle(idxUnls)

    # select labeled and unlabeled
    train, test = {}, {}

    # traing data index
    train['xl'] = idxLabs
    train['xu'] = idxUnls
    # test data
    test['x'] = [x[i][j] for i,j in idxTest]
    test['y'] = np.array([y[i][j] for i,j in idxTest], dtype=np.int32)

    return x, y, train, test

def de_duplicate(x):
    pass