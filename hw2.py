import numpy as np
import ray
from really import SampleManager
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
import gym
import os
from datetime import datetime

from really.utils import dict_to_dict_of_datasets


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

    tf.keras.backend.set_floatx('float64')

    model = DQN()

    # initialize
    ray.init(log_to_driver=False)
    manager = SampleManager(DQN, 'CartPole-v0',
                            num_parallel=3, total_steps=100)

    buffer_size = 2000
    epochs = 10
    saving_path = os.path.join(os.getcwd(), '/progress_hw2')
    saving_after = 5
    sample_size = 1000
    optim_batch_size = 8
    gamma = .98

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

        sample_dict = manager.sample(sample_size)
        print(f"collected data for: {sample_dict.keys()}")
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        print("optimizing... ")

        for state, action, reward, state_new, not_done in \
            zip(data_dict['state'],
                data_dict['action'],
                data_dict['reward'],
                data_dict['state_new'],
                data_dict['not_done']):
            #q_net = model(state)
            #q_target = reward + gamma * np.max(q_net)
            print("hey")
        #q_target = data_dict['reward'] + gamma * np.max(model(data_dict['state']))



