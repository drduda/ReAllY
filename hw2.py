import numpy as np
import ray
from really import SampleManager
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
import gym
import os
from datetime import datetime


class DQN(tf.keras.Model):
    def __init__(self, action_space=2, input_shape=2):
        super().__init__()
        self.action_space = action_space
        self.d1 = Dense(16, activation=LeakyReLU())
        self.d2 = Dense(32, activation=LeakyReLU())
        self.d3 = Dense(32, activation=LeakyReLU())
        self.dout = Dense(self.action_space, activation=None)

    def __call__(self, state):
        output = {}
        hidden = self.d1(state)
        hidden = self.d2(hidden)
        hidden = self.d3(hidden)
        q = self.dout(hidden)
        output["q_values"] = q
        return output


if __name__ == "__main__":

    model = DQN()

    # initialize
    ray.init(log_to_driver=False)
    manager = SampleManager(DQN, 'CartPole-v0',
                            num_parallel=5, total_steps=100)

    buffer_size = 2000
    epochs = 10
    saving_path = os.path.join(os.getcwd(), '/progress_hw2')
    saving_after = 5

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    manager.initialize_aggregator(
        path=saving_path,
        saving_after=saving_after,
        aggregator_keys=['loss', 'time_steps']
    )

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):
        print("collecting experience..")

        # Gives you state action reward trajetories
        data = manager.get_data()
        manager.store_in_buffer(data)
