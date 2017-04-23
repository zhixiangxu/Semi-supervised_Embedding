from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.recurrent import SimpleRNN, GRU

def forward(name, x):
    assert name in {'g50c','mnist',
                    'pavia0', 'paviaU1_l', 'paviaU1_u', 'paviaU2_l', 'paviaU2_u', 'paviaU3_l', 'paviaU3_u',
                    'indian0', 'indian1_l', 'indian1_u', 'indian2_l', 'indian2_u', 'indian3_l', 'indian3_u',
                    'houston0', 'houston1_l', 'houston1_u', 'houston2_l', 'houston2_u', 'houston3_l', 'houston3_u',
                    }, 'ERROR: wrong neural network name'
    layerlst = {# paviaU data
                'paviaU0':paviaU_model_1,
                'paviaU1_l':paviaU_model_1,
                'paviaU1_u':paviaU_model_1,
                'paviaU2_l':paviaU_model_2_label,
                'paviaU2_u':paviaU_model_2_unlabel,
                'paviaU3_l':paviaU_model_3_label,
                'paviaU3_u':paviaU_model_3_unlabel,
                # India data
                'indian0':indian_model_0,
                'indian1_l':indian_model_1,
                'indian1_u':indian_model_1,
                'indian2_l':indian_model_2_label,
                'indian2_u':indian_model_2_unlabel,
                'indian3_l':indian_model_3_label,
                'indian3_u':indian_model_3_unlabel,
                # Houston data
                'houston0':houston_model_1,
                'houston1_l':houston_model_1,
                'houston1_u':houston_model_1,
                'houston2_l':houston_model_2_label,
                'houston2_u':houston_model_2_unlabel,
                'houston3_l':houston_model_3_label,
                'houston3_u':houston_model_3_unlabel,
                # TOY
                'mnist':mnist_net,
                'g50c': g50c_net,}
    y = x
    for layer in layerlst[name]:
        y = layer(y)
    return y

###############################################

# Output

paviaU_model_1 = [
    Reshape((103, 1, 1)),

    Conv2D(nb_filter=32, nb_row=2, nb_col=1, init='glorot_normal'),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 1)),

    Conv2D(nb_filter=16, nb_row=2, nb_col=1, init='glorot_normal'),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 1)),

    Reshape((25 * 1 * 16,)),
    Dense(64, activation='sigmoid', init='glorot_normal'),
    Dense(9, init='glorot_normal'),
    Activation('softmax')
]

# Input

paviaU_embedding_space_2 = [
    Reshape((103, 1, 1)),

    Conv2D(nb_filter=32, nb_row=2, nb_col=1, init='glorot_normal'),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 1)),

    Reshape((51 * 1 * 32, )),
    Dense(9, init='glorot_normal'),
    Activation('sigmoid')
]

paviaU_model_2_label = paviaU_embedding_space_2 + [
    Dense(64, init='glorot_normal'),
    Activation('tanh'),
    Reshape((64, 1, 1)),

    Conv2D(nb_filter=16, nb_row=2, nb_col=1, init='glorot_normal'),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 1)),

    Reshape((31 * 1 * 16,)),
    Dense(64, activation='sigmoid', init='glorot_normal'),
    Dense(9, init='glorot_normal'),
    Activation('softmax')
]

paviaU_model_2_unlabel = paviaU_embedding_space_2

# Auxiliary

paviaU_embedding_space_3 = [
    Reshape((103, 1, 1)),

    Conv2D(nb_filter=32, nb_row=2, nb_col=1, init='glorot_normal'),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 1)),
]

paviaU_model_3_label = paviaU_embedding_space_3 + [
    Conv2D(nb_filter=16, nb_row=2, nb_col=1, init='glorot_normal'),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 1)),

    Reshape((25 * 1 * 16,)),
    Dense(64, activation='sigmoid', init='glorot_normal'),
    Dense(9, init='glorot_normal'),
    Activation('softmax')
]

paviaU_model_3_unlabel = paviaU_embedding_space_3 + [
    Reshape((51 * 1 * 32,)),
    Dense(9, init='glorot_normal'),
    Activation('sigmoid')
]

###############################################

# supervised

indian_model_0 = [
    Reshape((200, 1, 1)),

    Conv2D(nb_filter=32, nb_row=2, nb_col=1, init='lecun_uniform'),
    Activation('sigmoid'),
    MaxPooling2D(pool_size=(2, 1)),

    Conv2D(nb_filter=16, nb_row=2, nb_col=1, init='glorot_normal'),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 1)),

    Reshape((49 * 1 * 16,)),
    Dense(128, activation='sigmoid', init='glorot_normal'),
    Dense(16, init='glorot_normal'),
    Activation('softmax')
]

# Output

indian_model_1 = [
    Reshape((200, 1, 1)),

    Conv2D(nb_filter=32, nb_row=2, nb_col=1, init='lecun_uniform'),
    Activation('sigmoid'),
    MaxPooling2D(pool_size=(2, 1)),

    Conv2D(nb_filter=16, nb_row=2, nb_col=1, init='lecun_uniform'),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 1)),

    Reshape((49 * 1 * 16,)),
    Dense(128, activation='sigmoid', init='lecun_uniform'),
    Dense(16, init='glorot_normal'),
    Activation('softmax')
]

# Input

indian_embedding_space_2 = [
    Reshape((200, 1, 1)),

    Conv2D(nb_filter=32, nb_row=2, nb_col=1),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 1)),

    Conv2D(nb_filter=16, nb_row=2, nb_col=1),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 1)),

    Reshape((49 * 1 * 16, )),
    Dense(16, init='lecun_uniform'),
    Activation('tanh')
]

indian_model_2_label = indian_embedding_space_2 + [
    Dense(64, activation='sigmoid', init='lecun_uniform'),

    Dense(128, activation='tanh', init='lecun_uniform'),
    Dense(16, init='lecun_uniform'),
    Activation('softmax')
]

indian_model_2_unlabel = indian_embedding_space_2

# Auxiliary

indian_embedding_space_3 = [
    Reshape((200, 1, 1)),

    Conv2D(nb_filter=32, nb_row=2, nb_col=1, init='glorot_normal'),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 1)),
]

indian_model_3_label = indian_embedding_space_3 + [
    Conv2D(nb_filter=16, nb_row=2, nb_col=1, init='glorot_normal'),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 1)),

    Reshape((49 * 1 * 16,)),
    Dense(128, activation='sigmoid', init='glorot_normal'),
    Dense(16, init='glorot_normal'),
    Activation('softmax')
]

indian_model_3_unlabel = indian_embedding_space_3 + [
    Reshape((99 * 1 * 32,)),
    Dense(16, init='glorot_normal'),
    Activation('sigmoid')
]

###############################################

# Output

houston_model_1 = [
    Reshape((144,)),
    Dense(128, activation='relu'),
    #Dropout(0.25),
    Dense(15),
    Activation('softmax')
]

# houston_model_1 = [
#     Reshape((144, 1, 1)),
#
#     Conv2D(nb_filter=32, nb_row=2, nb_col=1, init='lecun_uniform'),
#     Activation('sigmoid'),
#     MaxPooling2D(pool_size=(2, 1)),
#
#     Conv2D(nb_filter=16, nb_row=2, nb_col=1, init='lecun_uniform'),
#     Activation('tanh'),
#     MaxPooling2D(pool_size=(2, 1)),
#
#     Reshape((35 * 1 * 16,)),
#     Dense(128, activation='sigmoid', init='lecun_uniform'),
#     Dense(15, init='glorot_normal'),
#     Activation('softmax')
# ]

# Input

houston_embedding_space_2 = [
    Reshape((144, 1, 1)),

    Conv2D(nb_filter=32, nb_row=2, nb_col=1),
    Activation('sigmoid'),
    MaxPooling2D(pool_size=(2, 1)),

    Reshape((71 * 1 * 16, )),
    Dense(128, init='lecun_uniform'),
    Activation('tanh'),
    Dense(15, init='lecun_uniform'),
    Activation('sigmoid')
]

houston_model_2_label = houston_embedding_space_2 + [
    Dense(128, activation='sigmoid', init='lecun_uniform'),
    Reshape((128, 1, 1)),

    Conv2D(nb_filter=8, nb_row=2, nb_col=1),
    Activation('sigmoid'),
    MaxPooling2D(pool_size=(2, 1)),
    Reshape((63 * 1 * 8,)),

    Dense(128, activation='tanh', init='lecun_uniform'),
    Dense(15, init='lecun_uniform'),
    Activation('softmax')
]

houston_model_2_unlabel = houston_embedding_space_2

# Auxiliary

houston_embedding_space_3 = [
    Reshape((144, 1, 1)),

    Conv2D(nb_filter=32, nb_row=2, nb_col=1, init='glorot_normal'),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 1)),
]

houston_model_3_label = houston_embedding_space_3 + [
    Conv2D(nb_filter=16, nb_row=2, nb_col=1, init='glorot_normal'),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 1)),

    Reshape((35 * 1 * 16,)),
    Dense(128, activation='sigmoid', init='glorot_normal'),
    Dense(15, init='glorot_normal'),
    Activation('softmax')
]

houston_model_3_unlabel = houston_embedding_space_3 + [
    Reshape((71 * 1 * 32,)),
    Dense(15, init='glorot_normal'),
    Activation('sigmoid')
]

###############################################

g50c_net = [
    Dense(128, activation='tanh', init='glorot_normal'),
    Dense(64, activation='tanh', init='glorot_normal'),
    Dense(1, activation='tanh', init='glorot_normal'),
]

mnist_net = [
    Conv2D(nb_filter=32, nb_row=2, nb_col=2, init='glorot_normal'),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(nb_filter=16, nb_row=2, nb_col=2, init='glorot_normal'),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 2)),

    Reshape((6*6*16,)),
    Dense(64, activation='tanh', init='glorot_normal'),
    Dense(10, activation='softmax', init='glorot_normal'),
]