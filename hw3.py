import numpy as np
import ray
from really import SampleManager
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
import os
from really.utils import dict_to_dict_of_datasets

"""
 first, disclaimer 
-> you can probably reuse some code of your deep q implementation
you probably want to implement a policy network which outputs 
a normal distribution so think about what keys you need for the 
output dictionary and also think about what method you need to 
sample actions (tip: look up 'continuous-normal-diagonal')
-if you want to be on-policy (thus no replay buffer) 
you can still use the manager's sampling method but just pass on 
from_buffer=False ->  manager.sample(sample_size, from_buffer=False)
-if you want to also use a state value estimate have two things in mind:
you can write one model that outputs both your state value and your policy 
with separate internal layers. 
However, if you want to optimize, you have to make sure to only optimize 
the part of your model you want to optimize 
(the policy part or the state value estimate part). 
You can for example achieve that by naming your layers in the model 
initialization and then filter model.trainable_variables according to 
these names to compute and apply your gradients separately
the state values returned by the manager can't be used to backpropagate 
directly (as the gradient flow is interrupted)
 so you might need to compute the state values again when optimizing
if you are on policy ant need log probabilities to train, be aware you cannot make use of the collected log prob values of the sample manager, as a) you might need gradient flow and b) if you change your policy the future probabilities will also change
"""

class ActorCritic(tf.keras.Model):
    def __init__(self, action_space=2, input_shape=8):
        super().__init__()

    def call(self, state):
        output = {}
        output["mu"] = None
        output["sigma"] = None
        return output

if __name__ == "__main__":

    tf.keras.backend.set_floatx('float64')

    # initialize
    ray.init(log_to_driver=False)
    manager = SampleManager(ActorCritic, 'LunarLanderContinuous-v2',
                            num_parallel=1, total_steps=100,
                            action_sampling_type="continous_normal_diagonal")

    buffer_size = 2000
    epochs = 100
    saving_path = os.getcwd() + "/hw3_results"
    saving_after = 5
    sample_size = 100
    optim_batch_size = 8
    gamma = .98
    update_interval = 4
    test_steps = 1000
    temperature = 1.5
    temperature_update = 0.98  # new_temp = old_temp*temp_update
    temperature_min = 0.5


    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]
    manager.initilize_buffer(buffer_size, optim_keys)

    agent = manager.get_agent()

    optimizer = tf.keras.optimizers.Adam()

    for e in range(epochs):
        sample_dict = manager.sample(sample_size, from_buffer=False)
        print(f"collected data for: {sample_dict.keys()}")