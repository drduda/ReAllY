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
    manager = SampleManager(DQN, 'CartPole-v0',
                            num_parallel=5, total_steps=100)

    pass