import numpy as np
import ray
from really import SampleManager
import tensorflow as tf
import gym
import os
from datetime import datetime

class DQN(tf.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, state):
        pass

    def set_weights(self, weights):
        pass

    def get_weights(self):
        pass

if __name__ == "__main__":

    model = DQN()

    # initialize
    ray.init(log_to_driver=False)
    manager = SampleManager(DQN, 'CartPole-v0',
                            num_parallel=5, total_steps=100)

    buffer_size = 2000
    epochs = 10

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):
        print("collecting experience..")

        # Gives you state action reward trajetories
        data = manager.get_data()
        manager.store_in_buffer(data)