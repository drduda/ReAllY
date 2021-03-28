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


class TD3Actor(tf.keras.layers.Layer):

    def __init__(self, action_dimension=2, min_action=-1, max_action=1):
        super(TD3Actor, self).__init__()
        self.action_dimension = action_dimension
        self.min_action = min_action
        self.max_action = max_action

        self.d1 = Dense(16, activation=LeakyReLU(), dtype=tf.float64)
        self.d2 = Dense(32, activation=LeakyReLU(), dtype=tf.float64)
        self.dout = Dense(self.action_dimension*2, activation=None, dtype=tf.float64)

    def call(self, inputs, training=None, mask=None):
        output = {}

        # pass through network
        hidden = self.d1(inputs)
        hidden = self.d2(hidden)
        dout = self.dout(hidden)

        # Clip mu to the possible actions spaces
        output['mu'] = tf.clip_by_value(dout[:, self.action_dimension:], self.min_action, self.max_action)
        output['sigma'] = tf.exp(dout[:, :self.action_dimension])

        return output

    def get_config(self):
        return super().get_config()


class TD3Critic(tf.keras.layers.Layer):

    def __init__(self):
        super(TD3Critic, self).__init__()

        self.d1 = Dense(16, activation=LeakyReLU(), dtype=tf.float64)
        self.d2 = Dense(32, activation=LeakyReLU(), dtype=tf.float64)
        self.dout = Dense(1, activation=None, dtype=tf.float64)

    def call(self, inputs, training=None, mask=None):
        output = {}

        # pass through network
        hidden = self.d1(inputs)
        hidden = self.d2(hidden)
        dout = self.dout(hidden)

        output['q_values'] = dout

        return output

    def get_config(self):
        return super().get_config()


class TD3Net(tf.keras.Model):
    def __init__(self, action_dimension=2, min_action=-1, max_action=1):
        super(TD3Net, self).__init__()
        self.action_dimension = action_dimension
        self.min_action = min_action
        self.max_action = max_action
        self.actor = TD3Actor(self.action_dimension, self.min_action, self.max_action)

        self.critic0 = TD3Critic()
        self.critic1 = TD3Critic()

    def call(self, state, **kwargs):
        output = {}

        actor_out = self.actor(state)
        output['mu'] = actor_out['mu']
        output['sigma'] = actor_out['sigma']

        # reparameterization trick
        action = output['mu'] + output['sigma'] * tf.random.normal(output['mu'].shape, 0., 1., dtype=tf.float64)
        # input for q network
        state_action = tf.concat([state, action], axis=-1, name='float error concat')

        output['q_values'] = tf.concat([
            self.critic0(state_action)['q_values'],
            self.critic1(state_action)['q_values']
        ], axis=-1)

        return output

    def get_config(self):
        return super(TD3Net, self).get_config()


if __name__ == "__main__":

    tf.keras.backend.set_floatx('float64')

    ray.init(log_to_driver=False)

    buffer_size = 200
    epochs = 2
    saving_path = os.getcwd() + "/smp_results"
    saving_after = 5
    sample_size = 15
    optim_batch_size = 8
    gamma = .99
    test_steps = 10
    update_interval = 4
    policy_delay = 2
    rho = .9

    manager = SampleManager(
        TD3Net,
        'LunarLanderContinuous-v2',
        num_parallel=(os.cpu_count() - 1),
        total_steps=150,
        action_sampling_type="continuous_normal_diagonal"
    )

    optim_keys = [
        'state',
        'action',
        'reward',
        'state_new',
        'not_done',
    ]

    manager.initilize_buffer(buffer_size, optim_keys)

    manager.initialize_aggregator(
        path=saving_path, saving_after=saving_after, aggregator_keys=["loss", "reward"]
    )

    agent = manager.get_agent()

    optimizer = tf.keras.optimizers.Adam(learning_rate=.0001)

    # fill buffer
    print("Filling buffer before training..")
    while True:
        # Check if buffer is already filled
        if len(manager.buffer.buffer[manager.buffer.keys[0]]) >= manager.buffer.size:
            break

        # Gives you state action reward trajectories
        data = manager.get_data()
        manager.store_in_buffer(data)

    target_agent = manager.get_agent()
    for e in range(epochs):
        if e % update_interval == 0:
            # Polyak averaging
            new_weights = rho * target_agent.get_weights + (1 - rho) * manager.get_agent().get_weights()
            target_agent.set_weights(new_weights)
        # off policy
        sample_dict = manager.sample(sample_size, from_buffer=True)
        print(f"collected data for: {sample_dict.keys()}")

        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        total_loss = 0
        for state, action, reward, state_new, not_done in \
            zip(data_dict['state'],
                data_dict['action'],
                data_dict['reward'],
                data_dict['state_new'],
                data_dict['not_done']):

            action_new = target_agent.flowing_log_prob(state, action)
            # add noise to action_new
            action_new = action_new + tf.random.normal(action_new.shape, 0., 1.)
            # clip action_new to action space
            action_new = tf.clip_by_value(
                action_new,
                manager.env_instance.action_space.low,
                manager.env_instance.action_space.high
            )

            # TODO: do we need tf.gather?
            state_action_new = tf.concat([state_new, action_new], axis=-1)
            q_values0 = target_agent.model.critic0(state_action_new)['q_values']
            q_values1 = target_agent.model.critic1(state_action_new)['q_values']
            q_values = tf.concat([q_values0, q_values1], axis=-1)
            q_targets = tf.reduce_min(q_values, axis=-1)
            critic_target = reward + gamma * tf.cast(not_done, tf.float64) * q_targets

            state_action = tf.concat([state, action], axis=-1)

            # update critic 0
            with tf.GradientTape() as tape:
                q_output = agent.model.critic0(state_action)['q_values']
                loss = tf.keras.losses.MSE(critic_target, q_output)

            total_loss += loss
            gradients = tape.gradient(loss, agent.model.critic0.trainable_variables)
            optimizer.apply_gradients(zip(gradients, agent.model.critic0.trainable_variables))

            # update critic 1
            with tf.GradientTape() as tape:
                q_output = agent.model.critic1(state_action)['q_values']
                loss = tf.keras.losses.MSE(critic_target, q_output)

            total_loss += loss
            gradients = tape.gradient(loss, agent.model.critic1.trainable_variables)
            optimizer.apply_gradients(zip(gradients, agent.model.critic1.trainable_variables))

            # update actor
            if e % policy_delay == 0:
                with tf.GradientTape() as tape:
                    # TODO: why sum?
                    action_prob = tf.reduce_sum(agent.flowing_log_prob(state, action), axis=-1)
                    state_action = tf.concat([state, action_prob], axis=-1)
                    q_val = agent.model.critic0(state_action)
                    actor_loss = - tf.reduce_mean(q_val)

                total_loss += actor_loss
                actor_gradients = tape.gradient(actor_loss, agent.model.actor.trainable_variables)
                optimizer.apply_gradients(zip(actor_gradients, agent.model.actor.trainable_variables))

            # Update agent
            manager.set_agent(agent.get_weights())
            agent = manager.get_agent()

        reward = manager.test(test_steps, evaluation_measure="reward")
        manager.update_aggregator(loss=total_loss, reward=reward)
        print(
            f"epoch ::: {e}  loss ::: {total_loss}   avg reward ::: {np.mean(reward)}"
        )

        if e % saving_after == 0:
            manager.save_model(saving_path, e)

    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)
