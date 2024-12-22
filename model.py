import tensorflow as tf
from tensorflow.keras.layers import Input, Permute, multiply, LayerNormalization, Add, Activation, BatchNormalization, \
    Reshape, Embedding, concatenate, Conv2D, Bidirectional, Dense, Flatten, RepeatVector, Dropout, Concatenate, LSTM, \
    GRU, Conv1D, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomUniform
from keras_multi_head import MultiHeadAttention
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import tensorflow as tf
from tensorflow.keras.layers import Input, Permute, LayerNormalization, Lambda, Attention, AveragePooling1D, add, \
    Activation, multiply, BatchNormalization, Reshape, Embedding, concatenate, Conv2D, Bidirectional, Dense, Flatten, \
    RepeatVector, Dropout, GRU, Conv1D
from tensorflow.keras import Model
from keras_multi_head import MultiHeadAttention
K = tf.keras.backend


def CRISPR_HNN():
    x1_input = Input(shape=(1, 23, 4))

    branch_0 = Conv2D(10, 1, strides=1, padding='same', use_bias=True,
                      name=None, trainable=True)(x1_input)
    branch_1 = Conv2D(10, 3, strides=1, padding='same', use_bias=True,
                      name=None, trainable=True)(x1_input)
    branch_2 = Conv2D(10, 5, strides=1, padding='same', use_bias=True,
                      name=None, trainable=True)(x1_input)
    branch_3 = Conv2D(10, 7, strides=1, padding='same', use_bias=True,
                      name=None, trainable=True)(x1_input)

    branches = [x1_input, branch_0, branch_1, branch_2, branch_3]

    mixed = concatenate(branches, axis=-1)
    mixed = Reshape((23, -1))(mixed)
    mixed = BatchNormalization()(mixed)

    x1 = MultiHeadAttention(head_num=4)([mixed, mixed, mixed])
    x1 = Add()([x1, mixed])
    x1 = LayerNormalization()(x1)

    x2 = MultiHeadAttention(head_num=4)([x1, x1, x1])
    x2 = Add()([x2, x1])
    x2 = LayerNormalization()(x2)

    x1_output = Bidirectional(GRU(128, return_sequences=True, input_shape=(23, -1)))(x2)

    x2_input = Input(shape=(24,))
    embedd = Embedding(input_dim=7, output_dim=64, input_length=24)(x2_input)
    embedd = Conv1D(32, 2, strides=1)(embedd)
    GRU_output = Bidirectional(GRU(128, return_sequences=True))(embedd)
    GRU_output = Bidirectional(GRU(128, return_sequences=True))(GRU_output)

    combined = concatenate([x1_output, GRU_output], axis=-1)
    flat_output = Flatten()(combined)

    dense_output = Dense(80, activation='relu')(flat_output)
    dense_output = Dense(20, activation='relu')(dense_output)
    dense_output = Dropout(0.35)(dense_output)

    output = Dense(1, activation='linear', name="output")(dense_output)

    model = Model(inputs=[x1_input, x2_input], outputs=[output])

    return model
