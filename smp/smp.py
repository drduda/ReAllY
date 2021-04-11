import logging
import os
import sys

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.activations import tanh

from smp_utils import reparam_action, parse, train_td3
from base_models import TD3Critic, MLPBase


def convert_mono_to_modular_state(mono_state):
    """
    :brief Helper function to convert a state obtained from the gym environment to four distinct states for each joint.
        The first leg is considered the *left* leg and the second one is considered the *right* one. The hull
        velocity and the ground contact are only given to the respective joints (head joint and knee joints) and
        is then expected to be passed indirectly to the other joints via messages.

    :param mono_state monolithic state from the gym environment
    """
    hull_angle = tf.reshape(mono_state[:, 0], [-1, 1])
    hull_speed = tf.reshape(mono_state[:, 1], [-1, 1])
    vel_x = tf.reshape(mono_state[:, 2], [-1, 1])
    vel_y = tf.reshape(mono_state[:, 3], [-1, 1])
    hip_angle_l = tf.reshape(mono_state[:, 4], [-1, 1])
    hip_speed_l = tf.reshape(mono_state[:, 5], [-1, 1])
    knee_angle_l = tf.reshape(mono_state[:, 6], [-1, 1])
    knee_speed_l = tf.reshape(mono_state[:, 7], [-1, 1])
    ground_contact_l = tf.reshape(mono_state[:, 8], [-1, 1])
    hip_angle_r = tf.reshape(mono_state[:, 9], [-1, 1])
    hip_speed_r = tf.reshape(mono_state[:, 10], [-1, 1])
    knee_angle_r = tf.reshape(mono_state[:, 11], [-1, 1])
    knee_speed_r = tf.reshape(mono_state[:, 12], [-1, 1])
    ground_contact_r = tf.reshape(mono_state[:, 13], [-1, 1])
    lidar = mono_state[:, 13:]

    """
    Joint state:
        angle
        speed
        foot ground contact
        head velocity x
        head velocity y
    """
    knee_state_l = tf.concat([
        knee_angle_l,
        knee_speed_l,
        ground_contact_l,
        tf.zeros_like(knee_angle_l),
        tf.zeros_like(knee_angle_l),
        tf.zeros_like(lidar)
    ], axis=-1)
    knee_state_r = tf.concat([
        knee_angle_r,
        knee_speed_r,
        ground_contact_r,
        tf.zeros_like(knee_angle_r),
        tf.zeros_like(knee_angle_r),
        tf.zeros_like(lidar)
    ], axis=-1)
    hip_state_l = tf.concat([
        hip_angle_l,
        hip_speed_l,
        tf.zeros_like(hip_angle_l),
        tf.zeros_like(hip_angle_l),
        tf.zeros_like(hip_angle_l),
        tf.zeros_like(lidar)
    ], axis=-1)
    hip_state_r = tf.concat([
        hip_angle_r,
        hip_speed_r,
        tf.zeros_like(hip_angle_r),
        tf.zeros_like(hip_angle_r),
        tf.zeros_like(hip_angle_r),
        tf.zeros_like(lidar)
    ], axis=-1)
    head_state = tf.concat([
        hull_angle,
        hull_speed,
        tf.zeros_like(hull_angle),
        vel_x,
        vel_y,
        lidar
    ], axis=-1)

    return knee_state_l, knee_state_r, hip_state_l, hip_state_r, head_state


def convert_modular_to_mono_action(action_hip_l, action_knee_l, action_hip_r, action_knee_r):
    return tf.concat([action_hip_l, action_knee_l, action_hip_r, action_knee_r], axis=1)


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
        knee_state_l, knee_state_r, hip_state_l, hip_state_r, head_state = convert_mono_to_modular_state(inputs)

        # Upwards policy
        # ugly hack to get the dimensions right
        up_messages_from_ground = tf.zeros_like(inputs)
        while up_messages_from_ground.shape[-1] < self.message_dimension:
            up_messages_from_ground = tf.concat([up_messages_from_ground, tf.zeros_like(inputs)], axis=-1)
        up_messages_from_ground = up_messages_from_ground[:, :self.message_dimension]

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


class TD3Net(tf.keras.Model):
    def __init__(self, action_dimension=2, min_action=-1, max_action=1, msg_dimension=1, hidden_units=128, fix_sigma=True):
        super(TD3Net, self).__init__()
        self.action_dimension = action_dimension
        self.min_action = min_action
        self.max_action = max_action
        self.msg_dimension = msg_dimension
        self.fix_sigma = fix_sigma
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


if __name__ == "__main__":
    args = parse(sys.argv[1:])
    train_td3(args, TD3Net, 1)
