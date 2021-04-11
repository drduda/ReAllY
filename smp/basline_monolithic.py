import logging
import os
import sys

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.activations import tanh

from smp_utils import reparam_action, parse, train_td3
from base_models import TD3Critic


class TD3Actor(tf.keras.layers.Layer):

    def __init__(self, action_dimension=2, min_action=-1, max_action=1, hidden_units=128, fix_sigma=True):
        super(TD3Actor, self).__init__()
        self.action_dimension = action_dimension
        self.min_action = min_action
        self.max_action = max_action
        self.fix_sigma = fix_sigma

        self.d1 = Dense(hidden_units, activation=LeakyReLU(), dtype=tf.float32)
        self.d2 = Dense(hidden_units, activation=LeakyReLU(), dtype=tf.float32)
        self.dout = Dense(self.action_dimension*2, activation=None, dtype=tf.float32)

    def call(self, inputs, training=None, mask=None):
        # pass through network
        hidden = self.d1(inputs)
        hidden = self.d2(hidden)
        dout = self.dout(hidden)

        mu = dout[:, :self.action_dimension]
        if self.fix_sigma:
            sigma = tanh(tf.ones_like(mu, dtype=tf.float32))
        else:
            sigma = tf.exp(tanh(dout[:, self.action_dimension:]))

        action = {
            'mu': mu,
            'sigma': sigma
        }

        return action

    def get_config(self):
        return super().get_config()


class TD3Net(tf.keras.Model):
    def __init__(self, action_dimension=2, min_action=-1, max_action=1, msg_dimension=None, hidden_units=128, fix_sigma=True):
        super(TD3Net, self).__init__()
        self.action_dimension = action_dimension
        self.min_action = min_action
        self.max_action = max_action
        self.actor = TD3Actor(self.action_dimension, self.min_action, self.max_action, hidden_units, fix_sigma=fix_sigma)

        self.critic0 = TD3Critic()
        self.critic1 = TD3Critic()

    def call(self, state, **kwargs):
        output = {}

        actor_out = self.actor(state)
        output['mu'] = actor_out['mu']
        output['sigma'] = actor_out['sigma']

        action = reparam_action(output, self.action_dimension, self.min_action, self.max_action)

        # input for q network
        state_action = tf.concat([state, action], axis=-1)

        q_values = tf.concat([
            self.critic0(state_action),
            self.critic1(state_action)
        ], axis=-1)

        output['q_values'] = tf.squeeze(tf.reduce_min(q_values, axis=-1))

        return output

    def get_config(self):
        return super(TD3Net, self).get_config()


if __name__ == "__main__":
    args = parse(sys.argv[1:])
    train_td3(args, TD3Net)
