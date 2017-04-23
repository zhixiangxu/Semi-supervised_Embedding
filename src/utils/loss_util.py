import tensorflow as tf
import numpy as np

Alpha = tf.constant(0.1)
Beta = 1.0 - Alpha
# Alpha is for mainfold kind of loss
# Beta is for cluster kind of loss

def supervised_loss(name):
    assert name in ('abs_hinge', 'hinge', 'abs_quadratic'), 'ERROR: wrong supervised loss name'
    supervised = {'hinge':hinge_loss,
                  'abs_hinge':abs_hinge_loss,
                  'abs_quadratic':abs_quadratic_loss}
    return supervised[name]

def graph_loss(name):
    assert name in ('SN', 'LE'), 'ERROR: wrong graph loss name'
    manifold = {'SN':Siamese_Networks,
                'LE':Laplacian_Eigenmaps}
    return manifold[name]

def cluster_loss(name):
    assert name in ('S3VM', 'EUDIV'), 'ERROR: wrong cluster loss name'
    cluster = {'S3VM':unlabeled_s3vm,
               'KLDA':kernelized_linear_discriminant_analysis,
               }
    return cluster[name]

# Supervised loss

def hinge_loss(y, yt):
    loss = tf.reduce_mean(tf.losses.hinge_loss(logits=y, labels=yt))
    return loss

def abs_hinge_loss(y, yt):
    m = tf.to_float(1.0)
    loss = tf.reduce_mean(tf.nn.relu(m - tf.abs(y * yt)))
    return loss

def abs_quadratic_loss(y, yt):
    loss = abs_hinge_loss(y, yt)
    return tf.square(loss)

def cross_entropy(y, yt):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=yt))
    return loss

# Manifold loss based on graph method(neighbor)

def Laplacian_Eigenmaps(yi, yj, wij):
    # attention that this distance is not Euclidean Distance !!!
    # this is square of Euclidean
    dist = tf.reduce_sum(tf.square(yi - yj), 1)
    w_ij = tf.to_float(wij)

    loss = tf.multiply(w_ij, dist)
    loss = tf.multiply(Alpha, loss)
    return tf.reduce_mean(loss)

def Siamese_Networks(yi, yj, wij):
    dist = tf.reduce_sum(tf.square(yi - yj), 1)
    margin = tf.constant(1.0)
    w_ij = tf.to_float(wij)

    match_loss = dist
    mismatch_loss = tf.square(tf.nn.relu(margin - dist))

    loss = tf.multiply(w_ij, match_loss) + tf.multiply((1 - w_ij), mismatch_loss)
    loss = tf.multiply(Alpha, loss)
    return tf.reduce_mean(loss)

# Cluster loss

def unlabeled_s3vm(y):
    loss = tf.reduce_mean(tf.nn.relu(1 - tf.abs(y)))
    loss = tf.multiply(Beta, loss)
    return loss

def embedding_mean(x, y):
    # x: labeled embedded data;
    # y: their labels/classes, better number instead of one-hot
    clst = list(set(y))
    means = {}
    for c in clst:
        key = str(c)
        means[key] = []
        for i in range(len(x)):
            if str(y[i]) == key:
                means[key] += [x[i]]
        means[key] = np.mean(np.array(means[key]), axis=0)

def between_scattering_matrix(mu1, mu2):
    between = tf.matmul(mu1, tf.transpose(mu2))
    return between

def within_scatter_mat(mdict, x):
    within = np.zeros((len(x[0]), len(x[0])))
    return

def kernelized_linear_discriminant_analysis(y):
    # here y is an embedding vector
    return