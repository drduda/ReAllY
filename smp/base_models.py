import logging
import os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, LayerNormalization


class MLPBase(tf.keras.layers.Layer):
    def __init__(self, out_dim, hidden_units):
        super(MLPBase, self).__init__()
        self.d1 = Dense(hidden_units, activation=LeakyReLU(), dtype=tf.float32)
        self.d2 = Dense(hidden_units, activation=LeakyReLU(), dtype=tf.float32)
        self.d3 = Dense(out_dim, activation=None, dtype=tf.float32)
        self.normalize = LayerNormalization()

    def call(self, inputs, training=None, mask=None):
        hidden = self.d1(inputs)
        hidden = self.d2(hidden)
        return self.d3(hidden)


class TD3Critic(tf.keras.layers.Layer):

    def __init__(self):
        super(TD3Critic, self).__init__()

        self.d1 = Dense(16, activation=LeakyReLU(), dtype=tf.float32)
        self.d2 = Dense(32, activation=LeakyReLU(), dtype=tf.float32)
        self.dout = Dense(1, activation=None, dtype=tf.float32)

    def call(self, inputs, training=None, mask=None):

        hidden = self.d1(inputs)
        hidden = self.d2(hidden)
        dout = self.dout(hidden)

        return dout

    def get_config(self):
        return super().get_config()
