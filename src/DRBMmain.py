import numpy as np
import tensorflow as tf
import scipy.io as sio
from sklearn import preprocessing
import random
import csv

from other.DRBM import Model
from utils import int_accuracy
from utils import dataset
import json
with open('./config/houston.json', 'r') as f:
    dconf = json.load(f)

dset = dataset(dconf)

lr = 0.01
num_hiddens = 2000

flags = tf.app.flags\
# Model input parameters
flags.DEFINE_integer("num_hidden", num_hiddens, "number of hidden units")
flags.DEFINE_integer("num_visible", dconf['data_shape'], "number of visible units")
flags.DEFINE_integer("num_classes", dconf['nb_class'], "number of classes")

# Learning hyperparameters
flags.DEFINE_integer("batch_size", 1, "batch size")
flags.DEFINE_integer("num_epochs", 1, "number of epochs")
flags.DEFINE_float("learning_rate", lr, "learning rate")
flags.DEFINE_float("alpha", 0.0, "generative objective weight")
flags.DEFINE_float("beta", 0.01, "semi-supervised objective weight")

# Debug
flags.DEFINE_string("model_name", "my_model", "name of the model")
flags.DEFINE_string("model_type", "semisup", "type of the model : [drbm, grbm, hybrid, semisup]")
flags.DEFINE_string("model_dir", "./debug/models/", "directory of saved checkpoints")
flags.DEFINE_string("logs_dir", "./debug/logs/", "directory to save the logs")
flags.DEFINE_integer("seed", 123, "random seed for python")

conf = flags.FLAGS

debug_results = 'debug_results.csv'

# learning_rates = [0.05, 0.01, 0.005, 0.001, 0.0005]
# num_hiddens = [100, 200, 600, 1000, 6000]

def main():
    conf.learning_rate = lr
    conf.num_hidden = num_hiddens
    conf.model_name = str(conf.model_type) + '_lr_' + str(lr) + '_nh_' + str(num_hiddens)
    my_scope = str(conf.model_type)
    conf.model_dir = './debug/' + dconf['name'] + '/' + conf.model_name + '/'

    with tf.Session() as sess:
        with tf.variable_scope(my_scope) as scope:
            model = Model(sess, conf)

            epoch_train_accuracy = np.zeros(conf.num_epochs)
            epoch_validation_accuracy = np.zeros(conf.num_epochs)
            epoch_test_accuracy = np.zeros(conf.num_epochs)

            # start_epoch = 16

            for epoch in range(conf.num_epochs):
                print('Epoch ', epoch, ' is starting...')
                model._load_model_from_checkpoint()

                print('Training...')
                avg_train_acc = model.train(sess, dset, lr, with_update=True)

                print('Average training accuracy : ', avg_train_acc)
                epoch_train_accuracy[epoch] = avg_train_acc

                model.inject_summary({'training_accuracy': avg_train_acc}, epoch)

                # print('Validation...')
                # avg_val_acc = model.train(sess, mnist.validation, with_update=False)
                #
                # print('Average validation accuracy : ', avg_val_acc)
                # epoch_validation_accuracy[epoch] = avg_val_acc
                #
                # model.inject_summary({'validation_accuracy': avg_val_acc}, epoch)

                stop_training = False

                if epoch > 5:
                    stop_training = True
                    for k in range(5):
                        if epoch_validation_accuracy[epoch - k] <= epoch_validation_accuracy[epoch - (k + 1)]:
                            stop_training = False

                print('Testing...')
                test_acc = model.test(sess, dset)

                print('Test result:')
                pred = model.predict(sess, dset)
                #print(int_accuracy(dset['y'], pred))
                for i in range(len(pred)):
                    print((pred[i], dset.test['y'][i]), end='')

                print('Testing accuracy : ', test_acc)
                print(list(set(list(pred))))
                epoch_test_accuracy[epoch] = test_acc

                print("Saving checkpoints...")
                save_path = model.save_model_to_checkpoint(epoch)
                print("Checkpoint succesfully saved...")

                with open(debug_results, 'a') as results_file:
                    writer = csv.writer(results_file, delimiter=',')
                    writer.writerow([conf.model_name, conf.model_type, epoch, conf.num_hidden,
                                     conf.learning_rate,
                                     avg_train_acc,
                                     test_acc, stop_training, save_path])

if __name__ == "__main__":
    main()
    print("\a")