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

    def call(self, state):
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

if __name__ == "__main__":

    # initialize
    ray.init(log_to_driver=False)
    manager = SampleManager(ActorCritic, 'LunarLanderContinuous-v2',
                            num_parallel=3, total_steps=150,
                            action_sampling_type="continuous_normal_diagonal",
                            #todo check if monte carlo is correct
                            #todo what about gamma??
                            returns=['monte_carlo', 'value_estimate', 'log_prob'])

    epochs = 200
    saving_path = os.getcwd() + "/hw3_results"
    saving_after = 5
    sample_size = 150
    optim_batch_size = 8
    gamma = .99
    test_steps = 1000
    # Factor of how much the new policy is allowed to differ from the old one
    epsilon = 0.2
    entropy_weight = 0.01

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "reward"]
    )

    agent = manager.get_agent()

    optimizer = tf.keras.optimizers.Adam()

    for e in range(epochs):
        sample_dict = manager.sample(sample_size, from_buffer=False)
        print(f"collected data for: {sample_dict.keys()}")

        # Shift value estimate by one to the left to get the value estimate of next state
        state_value = tf.squeeze(sample_dict['value_estimate'])
        state_value_new = tf.roll(state_value, -1, axis=0)
        not_done = tf.cast(sample_dict['not_done'], tf.bool)
        state_value_new = tf.where(not_done, state_value_new, 0)

        # Calculate advantate estimate q(s,a)-b(s)=r+v(s')-v(s)
        advantage_estimate = - state_value + sample_dict['reward'] + gamma * state_value_new
        sample_dict['advantage_estimate'] = advantage_estimate

        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)
        total_loss = 0

        for state, action, reward, state_new, not_done, mc, advantage_estimate, value_estimate, old_action_prob in \
            zip(data_dict['state'],
                data_dict['action'],
                data_dict['reward'],
                data_dict['state_new'],
                data_dict['not_done'],
                data_dict['monte_carlo'],
                data_dict['advantage_estimate'],
                data_dict['value_estimate'],
                data_dict['log_prob']):

            old_action_prob = tf.cast(old_action_prob, tf.float32)
            mc = tf.cast(mc, tf.float32)

            with tf.GradientTape() as tape:
                # Actor loss
                new_action_prob, entropy = agent.flowing_log_prob(state, action, return_entropy=True)
                new_action_prob = tf.reduce_sum(new_action_prob, axis=-1)
                entropy = tf.reduce_sum(entropy, axis=-1)
                actor_loss = (new_action_prob/old_action_prob) * advantage_estimate
                # Clipped Surrogate Objective is negative because of gradient ascent!
                actor_loss = tf.minimum(actor_loss, tf.clip_by_value(actor_loss, 1-epsilon, 1+epsilon))

                critic_loss = tf.reduce_mean((mc - agent.v(state))**2)

                loss = tf.reduce_mean(actor_loss + entropy_weight*entropy)
                # Signes are inverted because we technically use gradient descent
                loss = critic_loss - loss

            total_loss += loss
            gradients = tape.gradient(loss, agent.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))

            # Update the agent
            manager.set_agent(agent.get_weights())
            agent = manager.get_agent()

        reward = manager.test(test_steps, evaluation_measure="reward")
        manager.update_aggregator(loss=total_loss, reward=reward)
        # print progress
        print(
            f"epoch ::: {e}  loss ::: {total_loss}   avg env steps ::: {np.mean(reward)}"
        )

        if e % saving_after == 0:
            # you can save models
            manager.save_model(saving_path, e)

    # and load mmodels
    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)