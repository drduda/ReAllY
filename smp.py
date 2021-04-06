import logging
import os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, LayerNormalization
from tensorflow.keras.activations import tanh
import numpy as np
import gym
from gym.envs.box2d import BipedalWalker
import ray
from really import SampleManager
from really.utils import (
    dict_to_dict_of_datasets,
)
from copy import copy
import argparse


def convert_mono_to_modular_state(mono_state):
    """
    :brief Helper function to convert a state obtained from the gym environment to four distinct states for each joint.
        The first leg is considered the *left* leg and the second one is considered the *right* one. The hull
        velocity and the ground contact are only given to the respective joints (head joint and knee joints) and
        is then expected to be passed indirectly to the other joints via messages.

    :param mono_state monolithic state from the gym environment
    """
    hull_angle = mono_state[:, 0]
    hull_speed = mono_state[:, 1]
    vel_x = mono_state[:, 2]
    vel_y = mono_state[:, 3]
    hip_angle_l = mono_state[:, 4]
    hip_speed_l = mono_state[:, 5]
    knee_angle_l = mono_state[:, 6]
    knee_speed_l = mono_state[:, 7]
    ground_contact_l = mono_state[:, 8]
    hip_angle_r = mono_state[:, 9]
    hip_speed_r = mono_state[:, 10]
    knee_angle_r = mono_state[:, 11]
    knee_speed_r = mono_state[:, 12]
    ground_contact_r = mono_state[:, 13]

    """
    Joint state:
        angle
        speed
        foot ground contact
        head velocity x
        head velocity y
    """
    knee_state_l = tf.reshape(tf.concat([
        knee_angle_l,
        knee_speed_l,
        ground_contact_l,
        tf.zeros_like(knee_angle_l),
        tf.zeros_like(knee_angle_l)
    ], axis=-1), [-1, 5])
    knee_state_r = tf.reshape(tf.concat([
        knee_angle_r,
        knee_speed_r,
        ground_contact_r,
        tf.zeros_like(knee_angle_r),
        tf.zeros_like(knee_angle_r)
    ], axis=-1), [-1, 5])
    hip_state_l = tf.reshape(tf.concat([
        hip_angle_l,
        hip_speed_l,
        tf.zeros_like(hip_angle_l),
        tf.zeros_like(hip_angle_l),
        tf.zeros_like(hip_angle_l)
    ], axis=-1), [-1, 5])
    hip_state_r = tf.reshape(tf.concat([
        hip_angle_r,
        hip_speed_r,
        tf.zeros_like(hip_angle_r),
        tf.zeros_like(hip_angle_r),
        tf.zeros_like(hip_angle_r)
    ], axis=-1), [-1, 5])
    head_state = tf.reshape(tf.concat([
        hull_angle,
        hull_speed,
        tf.zeros_like(hull_angle),
        vel_x,
        vel_y
    ], axis=-1), [-1, 5])

    return knee_state_l, knee_state_r, hip_state_l, hip_state_r, head_state


def convert_modular_to_mono_action(action_hip_l, action_knee_l, action_hip_r, action_knee_r):
    return tf.concat([action_hip_l, action_knee_l, action_hip_r, action_knee_r], axis=1)


def reparam_action(act_dist, action_dimension, min_action, max_action):
    # re-parameterization
    action_out = act_dist['mu'] + act_dist['sigma'] * tf.random.normal([action_dimension], 0., 1., dtype=tf.float32)
    action_out = tf.clip_by_value(action_out, min_action, max_action)
    return action_out


class MLPBase(tf.keras.layers.Layer):
    def __init__(self, out_dim, hidden_units):
        super(MLPBase, self).__init__()
        self.d1 = Dense(hidden_units, activation=LeakyReLU(), dtype=tf.float32)
        self.d2 = Dense(hidden_units, activation=LeakyReLU(), dtype=tf.float32)
        self.d3 = Dense(out_dim, activation=None, dtype=tf.float32)
        self.normalize = LayerNormalization()

    def call(self, inputs, training=None, mask=None):
        hidden = self.d1(inputs)
        return self.d2(hidden)


class UpPolicy(MLPBase):
    # Upwards policy: message_up =policy_up(state, message_child)
    def __init__(self, message_dim, hidden_units):
        super(UpPolicy, self).__init__(message_dim, hidden_units)

    def __call__(self, state, message_child_1, message_child_2=None):
        if message_child_2 is None:
            message_child_2 = tf.zeros_like(message_child_1)
        inputs = tf.concat([state, message_child_1, message_child_2], axis=-1)
        return self.normalize(super().call(inputs))


class DownPolicy(MLPBase):
    def __init__(self, message_dim, action_dim, min_action, max_action, hidden_units, fix_sigma=True):
        super(DownPolicy, self).__init__(action_dim * 2 + message_dim * 2, hidden_units)
        self.action_dim = action_dim
        self.message_dim = message_dim
        self.min_action = min_action
        self.max_action = max_action
        self.fix_sigma = fix_sigma

        self.action_net = MLPBase(action_dim * 2, hidden_units)
        self.message_net = MLPBase(message_dim * 2, hidden_units)

    def __call__(self, message_up, message_down):
        inputs = tf.concat([message_up, message_down], axis=-1)
        action_out = self.action_net(inputs)
        message_out = self.message_net(inputs)

        action_mu = action_out[:, :self.action_dim]
        if self.fix_sigma:
            action_sigma = tanh(tf.ones_like(action_mu, dtype=tf.float32))
        else:
            action_sigma = tf.exp(tanh(action_out[:, self.action_dim:]))

        message_1 = self.normalize(message_out[:, :- self.message_dim])
        message_2 = self.normalize(message_out[:, self.message_dim:])

        action = {
            'mu': action_mu,
            'sigma': action_sigma
        }

        return action, message_1, message_2


class SMPActor(tf.keras.layers.Layer):
    def __init__(self, action_dimension, min_action, max_action, hidden_units, msg_dimension, fix_sigma=True):
        super(SMPActor, self).__init__()
        self.action_dimension = action_dimension
        self.min_action = min_action
        self.max_action = max_action
        self.message_dimension = msg_dimension
        self.number_modules = 5
        self.fix_sigma = fix_sigma

        # Upwards policy: message =policy_up(state, message1, message2)
        self.up_policy = UpPolicy(self.message_dimension, hidden_units)

        # Downwards policy: action, message1, message2 = policy_down(message_up, message,down)
        self.down_policy = DownPolicy(self.message_dimension, self.action_dimension, self.min_action, self.max_action, hidden_units,
                                      fix_sigma=self.fix_sigma)

    def call(self, inputs, training=None, mask=None):
        # discard lidar for now
        inputs = inputs[:, :-10]
        knee_state_l, knee_state_r, hip_state_l, hip_state_r, head_state = convert_mono_to_modular_state(inputs)

        # Upwards policy
        up_messages_from_ground = tf.zeros_like(inputs)[:, :self.message_dimension]

        # message_parent = policy_up(state, message_child)
        up_message_from_knee_l = self.up_policy(knee_state_l, up_messages_from_ground)
        up_message_from_knee_r = self.up_policy(knee_state_r, up_messages_from_ground)

        up_message_from_hip_l = self.up_policy(hip_state_l, up_message_from_knee_l)
        up_message_from_hip_r = self.up_policy(hip_state_r, up_message_from_knee_r)

        up_message_from_head = self.up_policy(head_state, up_message_from_hip_l, up_message_from_hip_r)

        # Downwards policy
        # action for head is not needed
        _, down_message_from_head_l, down_message_from_head_r = self.down_policy(
            up_message_from_head,
            tf.zeros_like(up_message_from_head)
        )

        # use different message channels for each side
        action_hip_l, down_message_from_hip_l, _ = self.down_policy(up_message_from_hip_l, down_message_from_head_l)
        action_hip_r, _, down_message_from_hip_r = self.down_policy(up_message_from_hip_r, down_message_from_head_r)

        action_knee_l, _, _ = self.down_policy(up_message_from_knee_l, down_message_from_hip_l)
        action_knee_r, _, _ = self.down_policy(up_message_from_knee_r, down_message_from_hip_r)

        action_out = {
            'mu': convert_modular_to_mono_action(
                action_hip_l['mu'],
                action_knee_l['mu'],
                action_hip_r['mu'],
                action_knee_r['mu']
            ),
            'sigma': convert_modular_to_mono_action(
                action_hip_l['sigma'],
                action_knee_l['sigma'],
                action_hip_r['sigma'],
                action_knee_r['sigma']
            ),
        }

        return action_out


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
    def __init__(self, action_dimension=2, min_action=-1, max_action=1, msg_dimension=1, hidden_units=128, fix_sigma=True):
        super(TD3Net, self).__init__()
        self.action_dimension = action_dimension
        self.min_action = min_action
        self.max_action = max_action
        self.msg_dimension = msg_dimension
        self.fix_sigma = fix_sigma
        print(msg_dimension)
        self.actor = SMPActor(self.action_dimension, self.min_action, self.max_action, hidden_units,
                              msg_dimension=self.msg_dimension, fix_sigma=fix_sigma)

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

def parse(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=150)
    parser.add_argument("--batch_size")
    parser.add_argument("--policy_noise")
    parser.add_argument("--msg_dim")
    parser.add_argument("--learning_rate")
    parser.add_argument("--hidden_units")
    parser.add_argument("--gamma")

    return parser.parse_args()

def main(args):

    args = parse(args)

    tf.keras.backend.set_floatx('float32')

    ray.init(log_to_driver=False)

    # hyper parameters
    buffer_size = 2 # 10e6 in their repo, not possible with our ram
    epochs = args.epochs
    saving_path = os.getcwd() + "/smp_results_test"
    saving_after = 5
    sample_size = 2
    optim_batch_size = args.batch_size
    gamma = args.gamma
    test_steps = 100 # 1000 in their repo
    update_interval = 4
    policy_delay = 2
    rho = .046
    policy_noise = args.policy_noise
    policy_noise_clip = .5
    msg_dim = 1 # 32 in their repo
    learning_rate = args.learning_rate

    env_test_instance = gym.make('BipedalWalker-v3')
    model_kwargs = {
        # action dimension for modular actions
        'action_dimension': 1,
        'min_action': copy(env_test_instance.action_space.low)[0],
        'max_action': copy(env_test_instance.action_space.high)[0],
        'msg_dimension': msg_dim,
        'fix_sigma': True,
        'hidden_units': args.hidden_units
    }
    del env_test_instance

    manager = SampleManager(
        TD3Net,
        'BipedalWalker-v3',
        num_parallel=(os.cpu_count() - 1),
        total_steps=150,
        action_sampling_type="continuous_normal_diagonal",
        is_tf=True,
        model_kwargs=model_kwargs
    )

    optim_keys = [
        'state',
        'action',
        'reward',
        'state_new',
        'not_done',
    ]

    manager.initialize_buffer(buffer_size, optim_keys)

    manager.initialize_aggregator(
        path=saving_path, saving_after=saving_after, aggregator_keys=["loss", "reward"]
    )

    agent = manager.get_agent()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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

        # cast values to float32 and create data dict
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
            action_new = action_new + tf.clip_by_value(
                tf.random.normal(action_new.shape, 0., policy_noise),
                -policy_noise_clip,
                policy_noise_clip
            )
            # clip action_new to action space
            action_new = tf.clip_by_value(
                action_new,
                manager.env_instance.action_space.low,
                manager.env_instance.action_space.high
            )

            # calculate target with double-Q-learning
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

            # update actor with delayed policy update
            if e % policy_delay == 0:
                with tf.GradientTape() as tape:
                    actor_output = agent.model.actor(state)
                    action = reparam_action(actor_output, agent.model.action_dimension,
                                            agent.model.min_action, agent.model.max_action)
                    state_action = tf.concat([state, action], axis=-1)
                    q_val = agent.model.critic0(state_action)
                    actor_loss = - tf.reduce_mean(q_val)

                total_loss += actor_loss
                actor_gradients = tape.gradient(actor_loss, agent.model.actor.trainable_variables)
                optimizer.apply_gradients(zip(actor_gradients, agent.model.actor.trainable_variables))

            # Update agent
            manager.set_agent(agent.get_weights())
            agent = manager.get_agent()

            if e % policy_delay == 0:
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


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
