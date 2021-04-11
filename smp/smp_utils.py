import logging
import os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import argparse
import gym
import ray
from really import SampleManager
from really.utils import (
    dict_to_dict_of_datasets
)
from copy import copy
import time


def reparam_action(act_dist, action_dimension, min_action, max_action):
    # re-parameterization
    action_out = act_dist['mu'] + act_dist['sigma'] * tf.random.normal([action_dimension], 0., 1., dtype=tf.float32)
    action_out = tf.clip_by_value(action_out, min_action, max_action)
    return action_out


def parse(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--policy_noise", type=float)
    parser.add_argument("--msg_dim", default=1, type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--hidden_units", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--buffer_size", default=2, type=int)
    parser.add_argument("--sample_size", default=2, type=int)
    parser.add_argument("--saving_dir", default="results_test", type=str)

    return parser.parse_args()


def save_args(args, saving_path):
    with open(saving_path + "/args.txt", "w") as f:
        f.write(args.__str__())


def train_td3(args, model, action_dimension=None):

    print(args)

    tf.keras.backend.set_floatx('float32')

    ray.init(log_to_driver=False)

    # hyper parameters
    buffer_size = args.buffer_size # 10e6 in their repo, not possible with our ram
    epochs = args.epochs
    saving_path = os.getcwd() + "/" + args.saving_dir
    saving_after = 5
    sample_size = args.sample_size
    optim_batch_size = args.batch_size
    gamma = args.gamma
    test_steps = 100 # 1000 in their repo
    policy_delay = 2
    rho = .046
    policy_noise = args.policy_noise
    policy_noise_clip = .5
    msg_dim = args.msg_dim # 32 in their repo
    learning_rate = args.learning_rate

    save_args(args, saving_path)

    env_test_instance = gym.make('BipedalWalker-v3')
    if action_dimension is None:
        action_dimension = copy(env_test_instance.action_space.shape[0])
    model_kwargs = {
        # action dimension for modular actions
        'action_dimension': action_dimension,
        'min_action': copy(env_test_instance.action_space.low)[0],
        'max_action': copy(env_test_instance.action_space.high)[0],
        'msg_dimension': msg_dim,
        'fix_sigma': True,
        'hidden_units': args.hidden_units
    }
    del env_test_instance

    manager = SampleManager(
        model,
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

    # track time while training
    timer = time.time()
    last_t = timer

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
                new_weights = list(rho * np.array(target_agent.get_weights())
                                   + (1. - rho) * np.array(agent.get_weights()))
                target_agent.set_weights(new_weights)

        reward = manager.test(test_steps, evaluation_measure="reward")
        manager.update_aggregator(loss=total_loss, reward=reward)
        print(
            f"epoch ::: {e}  loss ::: {total_loss}   avg reward ::: {np.mean(reward)}"
        )

        if e % saving_after == 0:
            manager.save_model(saving_path, e)

        # needed time and remaining time estimation
        current_t = time.time()
        time_needed = (current_t - last_t) / 60.
        time_remaining = (current_t - timer) / 60. / (e + 1) * (epochs - (e + 1))
        print('Finished epoch %d of %d. Needed %1.f min for this epoch. Estimated time remaining: %.1f min'
              % (e + 1, epochs, time_needed, time_remaining))
        last_t = current_t

    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)

    ray.shutdown()
