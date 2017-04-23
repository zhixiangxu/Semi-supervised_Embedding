import tensorflow as tf

def forward(name, x):
    assert name in {'DNN','CNN'}, 'ERROR: wrong neural network name'
    layerlst = {'DNN':neural_net, 
                'CNN':convolutional_net,}
    return layerlst[name](x)

def print_layer(epoch):
    val = DNN_var[2].eval()
    print('epoch', epoch)
    print(val[:10])

# oridinary neural net
DNN_var = [
    tf.Variable(tf.random_normal([50, 128], stddev=0.3)),
    tf.Variable(tf.zeros([128])),
    tf.Variable(tf.random_normal([128, 1], stddev=0.3)),
    tf.Variable(tf.zeros([1, 1]))
]

def neural_net(x):
    y = tf.nn.relu(tf.add(tf.matmul(x, DNN_var[0]), DNN_var[1]))
    y = tf.nn.tanh(tf.add(tf.matmul(y, DNN_var[2]), DNN_var[3]))
    return y

# convolutional net
def convolutional_net(x):
    pass