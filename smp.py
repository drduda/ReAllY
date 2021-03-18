import logging
import os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
import numpy as np
import gym
import ray
from really import SampleManager
from really.utils import (
    dict_to_dict_of_datasets,
)


class PPONet(tf.keras.Model):

    def __init__(self, action_dimension=2, min_action=-1, max_action=1):
        super().__init__()
        self.action_dimension = action_dimension
        self.min_action = min_action
        self.max_action = max_action

        # Actor net
        self.d1 = Dense(16, activation=LeakyReLU())
        self.d2 = Dense(32, activation=LeakyReLU())
        self.dout = Dense(self.action_dimension*2, activation=None)

        # Critic net
        self.v1 = Dense(16, activation=LeakyReLU())
        self.v2 = Dense(16, activation=LeakyReLU())
        self.vout = Dense(1, activation=None)

    def call(self, state, **kwargs):
        output = {}

        # Actor
        hidden = self.d1(state)
        hidden = self.d2(hidden)
        dout = self.dout(hidden)
        # Clip mu to the possible actions spaces
        output["mu"] = tf.clip_by_value(dout[:, self.action_dimension:], self.min_action, self.max_action)
        #todo check if really **e is needed
        output["sigma"] = tf.exp(dout[:, :self.action_dimension])

        # Critic
        hidden = self.v1(state)
        hidden = self.v2(hidden)
        vout = self.vout(hidden)
        output['value_estimate'] = vout

        return output

    def get_config(self):
        return super().get_config()


if __name__ == "__main__":

    ray.init(log_to_driver=False)
    manager = SampleManager(
        PPONet,
        'LunarLanderContinuous-v2',
        num_parallel=3,
        total_steps=150,
        action_sampling_type="continuous_normal_diagonal",
        returns=['monte_carlo', 'value_estimate', 'log_prob']
    )

    epochs = 2
    saving_path = os.getcwd() + "/smp_results"
    saving_after = 5
    sample_size = 150
    optim_batch_size = 8
    gamma = .99
    test_steps = 1000
    # Factor of how much the new policy is allowed to differ from the old one
    epsilon = 0.2
    entropy_weight = 0.01

    # add noise to a and based on that determine new r(s,a) and s'


    pass
