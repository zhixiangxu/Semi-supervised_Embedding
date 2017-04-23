import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils import dataset
from utils import supervised_loss, graph_loss, cluster_loss
from utils import forward
from utils import int_accuracy

import json
with open('./config/houston.json', 'r') as f:
    dconf = json.load(f)

def main():
    dset = dataset(dconf)
    # dset.set_batch_size([25, 25])
    
    input_dim = dconf['input_dim']
    output_dim = dconf['output_dim']
    lr = 1e-1
    # a random labeled example
    xl = tf.placeholder(tf.float32, input_dim)
    # a random neighbor
    xn = tf.placeholder(tf.float32, input_dim)
    # a random unlabeled example
    xu = tf.placeholder(tf.float32, input_dim)

    yl = forward(dconf['netName_label'], xl)
    yn = forward(dconf['netName_neighbor'], xn)
    yu = forward(dconf['netName_unlabel'], xu)
    yt = tf.placeholder(tf.float32, output_dim)

    loss_ = supervised_loss('abs_quadratic')(yl, yt)
    if not dconf['supervise only']:
        # add loss based on manifold neighbor
        loss_ += graph_loss('LE')(yl, yn, 1.0)
        loss_ += graph_loss('LE')(yl, yu, 0.0)
        # add loss based on cluster
        loss_ += cluster_loss('S3VM')(yn)

    opt = tf.train.AdagradOptimizer(lr)
    gvs = opt.compute_gradients(loss_)
    clipped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
    train_step = opt.apply_gradients(clipped_gvs)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    # actually this is the number of labeled samples used in training instead of epoch
    # because we cannot perform batch computation here
    nb_epoch = 100000
    for i in tqdm(range(nb_epoch)):
        xl_, xn_, xu_, yt_ = dset.pick()
        sess.run(train_step, feed_dict={xl:xl_, xn:xn_, xu:xu_, yt:yt_})

    # Test trained model
    with sess.as_default():
        if dconf['multi-label']:
            yl_ = sess.run(yl, feed_dict={xl: dset.test['x']})
            yl_ = np.argmax(np.array(yl_), axis=1)
            yt_ = dset.test['y']
            for metric in (int_accuracy,):
                m_name, m_value = metric(yl_, yt_)
                print(m_name, '%.6f'%m_value)
            for i in range(len(yt_)):
                print((yl_[i], yt_[i]), end='')
        else:
            for keras_metric in (int_accuracy, ):
                print(keras_metric(yt, yl).eval(feed_dict={xl: dset.test['x'], yt: dset.test['y']}))
            yp = sess.run(yl, feed_dict={xl: dset.test['x'], yt: dset.test['y']})
            for i in range(20):
                print(yp[i], dset.test['y'][i])

if __name__ == '__main__':
    main()