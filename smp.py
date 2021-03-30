import logging
import os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU
import numpy as np
import gym
from gym.envs.box2d import BipedalWalker
import ray
from really import SampleManager
from really.utils import (
    dict_to_dict_of_datasets,
)
from copy import copy


class TD3Actor(tf.keras.layers.Layer):

    def __init__(self, action_dimension=2, min_action=-1, max_action=1):
        super(TD3Actor, self).__init__()
        self.action_dimension = action_dimension
        self.min_action = min_action
        self.max_action = max_action

        self.d1 = Dense(16, activation=LeakyReLU(), dtype=tf.float32)
        self.d2 = Dense(32, activation=LeakyReLU(), dtype=tf.float32)
        self.dout = Dense(self.action_dimension*2, activation=None, dtype=tf.float32)

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
        action = output['mu'] + output['sigma'] * tf.random.normal([self.action_dimension], 0., 1., dtype=tf.float32)
        action = tf.clip_by_value(action, self.min_action, self.max_action)
        # input for q network
        state_action = tf.concat([state, action], axis=-1)

        q_values = tf.concat([
            self.critic0(state_action),
            self.critic1(state_action)
        ], axis=-1)

        if 'all_qs' in kwargs.keys() and kwargs['all_qs']:
            output['q_values'] = q_values
        else:
            output['q_values'] = tf.squeeze(tf.reduce_min(q_values, axis=-1))

        return output

    def get_config(self):
        return super(TD3Net, self).get_config()


if __name__ == "__main__":

    tf.keras.backend.set_floatx('float32')

    ray.init(log_to_driver=False)

    buffer_size = 2000
    epochs = 50
    saving_path = os.getcwd() + "/smp_monolithic_results"
    saving_after = 5
    sample_size = 15
    optim_batch_size = 8
    gamma = .99
    test_steps = 1000
    total_steps = 150
    update_interval = 4
    rho = .2

    env_test_instance = gym.make('BipedalWalker-v3')
    model_kwargs = {
        'action_dimension': copy(env_test_instance.action_space.shape[0]),
        'min_action': copy(env_test_instance.action_space.low),
        'max_action': copy(env_test_instance.action_space.high)
    }
    del env_test_instance

    manager = SampleManager(
        TD3Net,
        'BipedalWalker-v3',
        num_parallel=(os.cpu_count() - 1),
        total_steps=total_steps,
        action_sampling_type="continuous_normal_diagonal",
        model_kwargs=model_kwargs
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
    while len(manager.buffer.buffer[manager.buffer.keys[0]]) < manager.buffer.size:
        # Gives you state action reward trajectories
        data = manager.get_data()
        manager.store_in_buffer(data)

    target_agent = manager.get_agent()
    for e in range(epochs):
        # off policy
        sample_dict = manager.sample(sample_size, from_buffer=True)
        print(f"collected data for: {sample_dict.keys()}")

        sample_dict['state'] = tf.cast(sample_dict['state'], tf.float32)
        sample_dict['action'] = tf.cast(sample_dict['action'], tf.float32)
        sample_dict['reward'] = tf.cast(sample_dict['reward'], tf.float32)
        sample_dict['state_new'] = tf.cast(sample_dict['state_new'], tf.float32)
        sample_dict['not_done'] = tf.cast(sample_dict['not_done'], tf.float32)
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        total_loss = 0
        for state, action, reward, state_new, not_done in \
            zip(data_dict['state'],
                data_dict['action'],
                data_dict['reward'],
                data_dict['state_new'],
                data_dict['not_done']):

            action_new = target_agent.act(state_new)
            # add noise to action_new
            action_new = action_new + tf.random.normal(action_new.shape, 0., 1.)
            # clip action_new to action space
            action_new = tf.clip_by_value(
                action_new,
                manager.env_instance.action_space.low,
                manager.env_instance.action_space.high
            )

            state_action_new = tf.concat([state_new, action_new], axis=-1)
            q_values0 = target_agent.model.critic0(state_action_new)
            q_values1 = target_agent.model.critic1(state_action_new)
            q_values = tf.concat([q_values0, q_values1], axis=-1)
            q_targets = tf.squeeze(tf.reduce_min(q_values, axis=-1))
            critic_target = reward + gamma * not_done * q_targets

            state_action = tf.concat([state, action], axis=-1)

            # update critic 0
            with tf.GradientTape() as tape:
                q_output = agent.model.critic0(state_action)
                loss = tf.keras.losses.MSE(tf.squeeze(critic_target), tf.squeeze(q_output))

            total_loss += loss
            gradients = tape.gradient(loss, agent.model.critic0.trainable_variables)
            optimizer.apply_gradients(zip(gradients, agent.model.critic0.trainable_variables))

            # update critic 1
            with tf.GradientTape() as tape:
                q_output = agent.model.critic1(state_action)
                loss = tf.keras.losses.MSE(tf.squeeze(critic_target), tf.squeeze(q_output))

            total_loss += loss
            gradients = tape.gradient(loss, agent.model.critic1.trainable_variables)
            optimizer.apply_gradients(zip(gradients, agent.model.critic1.trainable_variables))

            # update actor
            if e % update_interval == 0:
                with tf.GradientTape() as tape:
                    actor_output = agent.model.actor(state)
                    action = actor_output['mu'] + actor_output['sigma'] * tf.random.normal(actor_output['mu'].shape, 0., 1., dtype=tf.float32)
                    action = tf.clip_by_value(action, agent.model.min_action, agent.model.max_action)
                    state_action = tf.concat([state, action], axis=-1)
                    q_val = agent.model.critic0(state_action)
                    actor_loss = - tf.reduce_mean(q_val)

                total_loss += actor_loss
                actor_gradients = tape.gradient(actor_loss, agent.model.actor.trainable_variables)
                optimizer.apply_gradients(zip(actor_gradients, agent.model.actor.trainable_variables))

            # Update agent
            manager.set_agent(agent.get_weights())
            agent = manager.get_agent()

            if e % update_interval == 0:
                # Polyak averaging
                new_weights = [rho * target_agent.get_weights()[i] + (1 - rho) * manager.get_agent().get_weights()[i]
                               for i in range(len(agent.get_weights()))]
                target_agent.set_weights(new_weights)

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
