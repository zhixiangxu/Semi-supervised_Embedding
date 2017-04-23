"""Code for training the Discriminative Restricted Boltzmann Machine.

This script is to be run with the following command-line options with their
respective arguments:
    -d  Dataset list file (See examples/datasets).
    -m  Model configuration file (See examples/models).
    -o  Optimizer configuration file (See examples/optimizers).

yangminz:
Fix some code lines for semi-supervised training
"""

import tensorflow as tf
import numpy as np
import random
import scipy.misc
from logging import getLogger
import datetime
import dateutil.tz
from datetime import date
from tqdm import tqdm

import os
import sys
import urllib
import pprint
import tarfile

import scipy.misc

import csv
import os

class Model:
    def __init__(self, sess, conf, checkpoint_fname=None):
        self.sess = sess

        # Model input parameters
        self.num_hidden = conf.num_hidden
        self.num_visible = conf.num_visible
        self.num_classes = conf.num_classes

        # Learning hyper-parameters
        self.hparams = {}
        self.hparams['batch_size'] = conf.batch_size
        self.hparams['num_epochs'] = conf.num_epochs
        self.hparams['learning_rate'] = conf.learning_rate
        # Generative objective weight
        self.hparams['alpha'] = conf.alpha
        self.hparams['beta'] = conf.beta

        # Internal stuff
        self.seed = conf.seed

        # Logging and saving parameters
        self.model_name = conf.model_name
        self.logs_dir = conf.logs_dir
        self.model_type = conf.model_type
        self.model_dir = conf.model_dir

        self._build_model()

        with tf.variable_scope('summary'):
            scalar_summary_tags = ['training_accuracy', 'validation_accuracy']

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])

            self.writer = tf.summary.FileWriter(os.path.join(self.model_dir, 'logs'), self.sess.graph)

        self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver()

        tf.global_variables_initializer().run()

        self._load_model_from_checkpoint(checkpoint_fname)

    def _load_model_from_checkpoint(self, checkpoint_fname=None):
        print(" [*] Loading checkpoints...")

        if checkpoint_fname is not None:
            self.saver.restore(self.sess, checkpoint_fname)
            return True
        else:
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                fname = os.path.join(self.model_dir, ckpt_name)
                self.saver.restore(self.sess, fname)
                print(" [*] Load SUCCESS: %s" % fname)
                return True
            else:
                print(" [!] Load FAILED: %s" % self.model_dir)
            return False

    def save_model_to_checkpoint(self, step):
        print(" [*] Saving checkpoints...")
        model_name = type(self).__name__

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        return self.saver.save(self.sess, self.model_dir, global_step=step)

    def _build_model(self, with_init=True):
        self.model = {}
        with tf.variable_scope(self.model_name):
            self.model['X'], self.model['XU'], self.model['Y'], self.model['learning_rate'] = self._create_placeholders()

            m = np.max([self.num_hidden, self.num_classes, self.num_visible])

            self.model['U'], self.model['W'], self.model['b'], self.model['c'], self.model['d'] = self._create_matrices(1. / np.sqrt(m))

            # Defines the internal
            self._construct_internal_variables()

            self.model['d_U'], self.model['d_W'], self.model['d_b'], self.model['d_c'], self.model['d_d'] = self._define_gradients()

            self.updates = [self.model['U'].assign_add(self.model['learning_rate'] * self.model['d_U']),
                            self.model['W'].assign_add(self.model['learning_rate'] * self.model['d_W']),
                            self.model['b'].assign_add(self.model['learning_rate'] * self.model['d_b']),
                            self.model['c'].assign_add(self.model['learning_rate'] * self.model['d_c']),
                            self.model['d'].assign_add(self.model['learning_rate'] * self.model['d_d'])]

            self.predicted_y = tf.argmax(self.model['p_y_all_given_x'], 1)

            self.ground_truth = tf.argmax(self.model['Y'], 1)

            self.correct_prediction = tf.equal(self.predicted_y, self.ground_truth, name='correct_prediction')
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), keep_dims=True,
                                           name='accuracy')

    def _construct_internal_variables(self):
        # Tensor of shape (num_classes x num_classes)
        self.model['Y_all_classes'] = tf.diag(tf.ones(self.num_classes, 1), name='all_the_classes')

        # tensor of shape (num_hidden x num_classes)
        self.model['U_all_y'] = tf.matmul(self.model['U'], self.model['Y_all_classes'])

        self.model['WX'], self.model['O_all'], self.model['positive_part'], self.model[
            'p_y_all_given_x'] = self._give_p_all_y_given_x()

        if self.model_type in ('drbm', 'hybrid', 'semisup'):
            # Calculate p(y|x) for concrete x
            # Result : p_y_given_x (None x 1)
            self.model['p_y_given_x'] = tf.reshape(
                tf.reduce_sum(tf.matmul(self.model['p_y_all_given_x'], self.model['Y']), 1), [-1, 1])

            # Training part

            # Calculate UY
            # Result : U (None x num_hidden)
            self.model['UY'] = tf.matmul(self.model['Y'], tf.transpose(self.model['U']), name='UY')

            self.model['O'] = self.model['WX'] + self.model['UY']

            # O_sigma: (None x num_hidden)
            self.model['O_sigma'] = tf.sigmoid(self.model['O'])

            # O_sigma_all_Y : (None x num_hidden x num_classes)
            self.model['O_sigma_all_Y'] = tf.sigmoid(self.model['O_all'])

            # O_sigma_all_Y_p : (None x num_hidden x num_classes)
            #
            # YANGMINZ: CHANGE BACKUP
            #
            # self.model['O_sigma_all_Y_p'] = tf.matmul(self.model['O_sigma_all_Y'], tf.tile(
            #     tf.reshape(self.model['p_y_all_given_x'], [-1, 1, self.num_classes]), [1, self.num_hidden, 1]))
            #
            self.model['O_sigma_all_Y_p'] = tf.matmul(self.model['O_sigma_all_Y'], tf.tile(
                tf.reshape(self.model['p_y_all_given_x'], [-1, 1, self.num_classes]), [1, self.num_classes, 1]))

    def _give_p_all_y_given_x(self):
        # Tensor of shape (None x num_hidden)
        WX = tf.matmul(self.model['X'], tf.transpose(self.model['W']), name='WX')

        # WX + c for all batches
        # Result : O (None x num_hidden)
        O = WX + tf.transpose(self.model['c'])

        # Resulted O:
        # Result: O (None x num_hidden x num_classes)
        O = tf.reshape(O, [-1, self.num_hidden, 1]) + tf.reshape(self.model['U_all_y'],
                                                                 [1, self.num_hidden, self.num_classes])

        # First term in log p(y|x) which is calculated for each x in the batch
        # Result : first_term (1 x num_classes)
        first_term = tf.matmul(tf.transpose(self.model['d']), self.model['Y_all_classes'])

        # Second term in log p(y|x) which is calculated for each x in the batch
        # Result : second_term (None x num_classes)
        second_term = tf.reduce_sum(tf.nn.softplus(O), 1)

        # Positive part of log p(y|x)
        # Result: positive_part (None x num_classes)
        positive_part = first_term + second_term

        # Use the softmax to calculate the probabilities:
        # Result: p_y_all_given_x (None x num_classes)
        p_y_all_given_x = tf.nn.softmax(positive_part)

        return WX, O, positive_part, p_y_all_given_x

    def _define_gradients(self):
        d_U = tf.zeros([self.num_hidden, self.num_classes], dtype=tf.float32, name='d_U')
        d_W = tf.zeros([self.num_hidden, self.num_visible], dtype=tf.float32, name='d_W')
        d_b = tf.zeros([self.num_visible, 1], dtype=tf.float32, name='d_b')
        d_c = tf.zeros([self.num_hidden, 1], dtype=tf.float32, name='d_c')
        d_d = tf.zeros([self.num_classes, 1], dtype=tf.float32, name='d_d')

        if self.model_type in ('grbm', 'hybrid', 'semisup'):
            # Generative gradients
            d_U_gen, d_W_gen, d_b_gen, d_c_gen, d_d_gen = self._calc_generative_grads(self.model['Y'], self.model['X'])
            # Semi-supervised gradients
            d_U_unsup, d_W_unsup, d_b_unsup, d_c_unsup, d_d_unsup = self._calc_unsupervised_grads(self.model['XU'])

            if self.model_type == 'grbm':
                d_U = d_U_gen
                d_W = d_W_gen
                d_b = d_b_gen
                d_c = d_c_gen
                d_d = d_d_gen
            elif self.model_type == 'hybrid':
                d_U = d_U + self.hparams['alpha'] * d_U_gen
                d_W = d_W + self.hparams['alpha'] * d_W_gen
                d_b = d_b + self.hparams['alpha'] * d_b_gen
                d_c = d_c + self.hparams['alpha'] * d_c_gen
                d_d = d_d + self.hparams['alpha'] * d_d_gen
            elif self.model_type == 'semisup':
                d_U = d_U + self.hparams['alpha'] * d_U_gen
                d_W = d_W + self.hparams['alpha'] * d_W_gen
                d_b = d_b + self.hparams['alpha'] * d_b_gen
                d_c = d_c + self.hparams['alpha'] * d_c_gen
                d_d = d_d + self.hparams['alpha'] * d_d_gen

                d_U = d_U + self.hparams['beta'] * d_U_unsup
                d_W = d_W + self.hparams['beta'] * d_W_unsup
                d_b = d_b + self.hparams['beta'] * d_b_unsup
                d_c = d_c + self.hparams['beta'] * d_c_unsup
                d_d = d_d + self.hparams['beta'] * d_d_unsup

        if self.model_type in ('drbm', 'hybrid', 'semisup'):
            # Discriminative gradients
            # # d_U: (num_hidden x num_classes)
            dU_left = tf.matmul(tf.transpose(self.model['O_sigma']), self.model['Y'])
            dU_right = tf.matmul(tf.transpose(tf.reduce_sum(self.model['O_sigma_all_Y_p'], 2)), self.model['Y'])
            d_U_disc = tf.div(dU_left - dU_right, self.hparams['batch_size'])
            d_U_disc = tf.reshape(d_U_disc, [self.num_hidden, self.num_classes])

            # d_W : (num_hidden x num_visible)
            dW_left = tf.matmul(tf.transpose(self.model['O_sigma']), self.model['X'])
            dW_right = tf.matmul(tf.transpose(tf.reduce_sum(self.model['O_sigma_all_Y_p'], 2)), self.model['X'])

            d_W_disc = tf.div(dW_left - dW_right, self.hparams['batch_size'])
            d_W_disc = tf.reshape(d_W_disc, [self.num_hidden, self.num_visible])

            # d_c : (num_hidden x 1)
            dc_left = tf.reduce_sum(self.model['O_sigma'], 0)
            dc_right = tf.reduce_sum(tf.reduce_sum(self.model['O_sigma_all_Y_p'], 2), 0)
            d_c_disc = tf.div(dc_left - dc_right, self.hparams['batch_size'])
            d_c_disc = tf.reshape(d_c_disc, [self.num_hidden, 1])

            # d_d : (num_classes x 1)
            d_d_disc = tf.div(tf.reduce_sum(self.model['Y'] - self.model['p_y_all_given_x'], 0),
                              self.hparams['batch_size'])
            d_d_disc = tf.reshape(d_d_disc, [self.num_classes, 1])

            d_U = d_U + d_U_disc
            d_W = d_W + d_W_disc
            d_c = d_c + d_c_disc
            d_d = d_d + d_d_disc

        return d_U, d_W, d_b, d_c, d_d

    def _calc_generative_grads(self, y, x):
        y0, x0, h0, y1, x1, h1 = self._gibbs_sampling_step(y, x)

        h0 = tf.reshape(h0, [-1, self.num_hidden, 1])
        y0 = tf.reshape(y0, [-1, self.num_classes, 1])
        x0 = tf.reshape(x0, [-1, self.num_visible, 1])
        h1 = tf.reshape(h1, [-1, self.num_hidden, 1])
        y1 = tf.reshape(y1, [-1, self.num_classes, 1])
        x1 = tf.reshape(x1, [-1, self.num_visible, 1])

        d_U_gen = tf.reduce_mean(tf.matmul(h0, y0, transpose_b=True) - tf.matmul(h1, y1, transpose_b=True), 0)
        d_W_gen = tf.reduce_mean(tf.matmul(h0, x0, transpose_b=True) - tf.matmul(h1, x1, transpose_b=True), 0)
        #
        # YANGMINZ: CHANGE BACKUP
        #
        # d_U_gen = tf.reduce_mean(tf.matmul(h0, y0, adj_y=True) - tf.matmul(h1, y1, adj_y=True), 0)
        # d_W_gen = tf.reduce_mean(tf.matmul(h0, x0, adj_y=True) - tf.matmul(h1, x1, adj_y=True), 0)

        d_b_gen = tf.reduce_sum(x0 - x1, 0)
        d_c_gen = tf.reduce_sum(h0 - h1, 0)
        d_d_gen = tf.reduce_sum(y0 - y1, 0)

        return d_U_gen, d_W_gen, d_b_gen, d_c_gen, d_d_gen

    def _gibbs_sampling_step(self, y, x):
        # Positive phase
        y0 = y
        x0 = x
        h0 = tf.nn.sigmoid(tf.transpose(
            self.model['c'] + tf.matmul(self.model['W'], tf.transpose(x0)) + tf.matmul(self.model['U'],
                                                                                       tf.transpose(y0))))

        # Negative phase
        h0new = self._sample_h(h0)
        y1 = self._sample_y(h0new)
        x1 = self._sample_x(h0new)
        h1 = tf.nn.sigmoid(tf.transpose(
            self.model['c'] + tf.matmul(self.model['W'], tf.transpose(x1)) + tf.matmul(self.model['U'],
                                                                                       tf.transpose(y1))))

        return y0, x0, h0, y1, x1, h1

    def _calc_unsupervised_grads(self, x):
        x0, h0, x1, h1 = self._gibbs_un_sampling_step(x)

        h0 = tf.reshape(h0, [-1, self.num_hidden, 1])
        x0 = tf.reshape(x0, [-1, self.num_visible, 1])
        h1 = tf.reshape(h1, [-1, self.num_hidden, 1])
        x1 = tf.reshape(x1, [-1, self.num_visible, 1])

        # no gradient for U, but use zero to take place here
        d_U_unsup = tf.constant(0.0)
        d_W_unsup = tf.reduce_mean(tf.matmul(h0, x0, transpose_b=True) - tf.matmul(h1, x1, transpose_b=True), 0)

        d_b_unsup = tf.reduce_sum(x0 - x1, 0)
        d_c_unsup = tf.reduce_sum(h0 - h1, 0)
        d_d_unsup = tf.constant(0.0)
        return d_U_unsup, d_W_unsup, d_b_unsup, d_c_unsup, d_d_unsup

    def _gibbs_un_sampling_step(self, x):
        # Positive phase
        x0 = x
        h0 = tf.nn.sigmoid(tf.transpose(
            self.model['c'] + tf.matmul(self.model['W'], tf.transpose(x0))))

        # Negative phase
        h0new = self._sample_h(h0)
        x1 = self._sample_x(h0new)
        h1 = tf.nn.sigmoid(tf.transpose(
            self.model['c'] + tf.matmul(self.model['W'], tf.transpose(x1))))

        return x0, h0, x1, h1

    def _sample_prob(self, probs, size):
        rand = tf.random_uniform([self.hparams['batch_size'], size], minval=0.0, maxval=1.0, dtype=tf.float32)
        return tf.cast(rand < probs, tf.float32)

    def _sample_h(self, h_prob):
        return self._sample_prob(h_prob, self.num_hidden)

    def _sample_y(self, h):
        yprob = tf.nn.softmax(tf.transpose(self.model['d'] + tf.matmul(tf.transpose(self.model['U']), tf.transpose(h))),
                              dim=-1)
        squeezed_y = tf.squeeze(tf.one_hot(tf.multinomial(yprob, 1), self.num_classes), [1])
        return tf.matmul(squeezed_y, self.model['Y_all_classes'])

    def _sample_x(self, h):
        xprob = tf.nn.sigmoid(tf.transpose(self.model['b'] + tf.matmul(tf.transpose(self.model['W']), tf.transpose(h))))
        return self._sample_prob(xprob, self.num_visible)

    def _create_placeholders(self):
        X = tf.placeholder(tf.float32, [None, self.num_visible])

        XU = tf.placeholder(tf.float32, [None, self.num_visible])

        Y = tf.placeholder(tf.float32, [None, self.num_classes])

        learning_rate = tf.placeholder(tf.float32)

        return X, XU, Y, learning_rate

    def _create_matrices(self, m_sqrt):
        U = tf.get_variable('U', [self.num_hidden, self.num_classes], tf.float32,
                            tf.random_uniform_initializer(minval=-m_sqrt, maxval=m_sqrt, seed=self.seed,
                                                          dtype=tf.float32), None)

        W = tf.get_variable('W', [self.num_hidden, self.num_visible], tf.float32,
                            tf.random_uniform_initializer(minval=-m_sqrt, maxval=m_sqrt, seed=self.seed,
                                                          dtype=tf.float32), None)

        b = tf.get_variable('b', [self.num_visible, 1], tf.float32,
                            tf.zeros_initializer(), None)

        c = tf.get_variable('c', [self.num_hidden, 1], tf.float32,
                            tf.zeros_initializer(), None)

        d = tf.get_variable('d', [self.num_classes, 1], tf.float32,
                            tf.zeros_initializer(), None)

        return U, W, b, c, d

    def inject_summary(self, tag_dict, step):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })

        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, step)

    def _get_timestamp(self):
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        return now.strftime('%Y_%m_%d_%H_%M_%S')

    def train(self, sess, dset, lr, with_update=False, debug_learning_rate=None):
        num_batches = int(dset.length[0] / self.hparams['batch_size'])
        #num_batches = 1000
        print(" [$] number of batches:", num_batches)

        accuracies = np.zeros(num_batches)

        for i in tqdm(range(num_batches)):
        #for i in range(num_batches):
            x, y, xu = dset.next_batch(self.hparams['batch_size'])

            if with_update == True:
                if debug_learning_rate is not None:
                    _, accuracy = self.sess.run([self.updates, self.accuracy],
                                                feed_dict={self.model['X']: x,
                                                           self.model['XU']: xu,
                                                           self.model['Y']: y,
                                                           self.model['learning_rate']: debug_learning_rate})
                else:
                    _, accuracy = self.sess.run([self.updates, self.accuracy],
                                                feed_dict={self.model['X']: x,
                                                           self.model['XU']: xu,
                                                           self.model['Y']: y,
                                                           self.model['learning_rate']: lr})

            else:
                accuracy = self.sess.run([self.accuracy], feed_dict={self.model['X']: x,
                                                                     self.model['XU']: xu,
                                                                     self.model['Y']: y})

            accuracies[i] = accuracy[0]

        return np.mean(accuracies)

    def test(self, sess, dset):
        x_test = dset.test['x']
        y_test = dset.OneHot(dset.test['y'], dset.pick_reshape['y'][1])
        accuracy = self.sess.run([self.accuracy],
                                 feed_dict={self.model['X']: x_test, self.model['Y']: y_test})
        return accuracy[0]
    
    def predict(self, sess, dset):
        x_test = dset.test['x']
        pred = self.sess.run(self.predicted_y, feed_dict={self.model['X']: x_test})
        return pred

    def _binarise(self, x):
        return (x > 0).astype(np.float32)
