from utils import dataset
import matplotlib.pyplot as plt
import numpy as np
import json

with open('./config/houston.json', 'r') as f:
    dconf = json.load(f)
dset = dataset(dconf)
a = dconf['shape'][2]
d = dset.x.reshape([dconf['shape'][0] * dconf['shape'][1], a])
l = dset.y.flatten()
print(set(l))

def draw_class(c):
    x = np.arange(a)
    fig, ax = plt.subplots()
    for i in range(len(l)):
        if int(l[i]) == c:
            line, = ax.plot(x, d[i])
    plt.savefig('/home/yangminz/Semi-supervised_Embedding/image/%s%02d.png'%(dconf['name'], c))

def main():
    draw_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    for i in draw_list:
        draw_class(i)

if __name__ == '__main__':
    main()