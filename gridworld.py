import gym
import numpy as np
import ray
from really import SampleManager
from gridworlds import GridWorld
import os
from really.utils import (
    dict_to_dict_of_datasets,
)  # convenient function for you to create tensorflow datasets

"""
Your task is to solve the provided Gridword with tabular Q learning!
In the world there is one place where the agent cannot go, the block.
There is one terminal state where the agent receives a reward.
For each other state the agent gets a reward of 0.
The environment behaves like a gym environment.
Have fun!!!!

"""


class TabularQ(object):
    def __init__(self, h, w, action_space):
        self.action_space = action_space

        ## # TODO:
        self.h = h
        self.w = w
        self.q_vals = np.zeros((self.h, self.w, self.action_space))

    def __call__(self, state):
        ## # TODO:
        output = {}
        output["q_values"] = np.random.normal(size=(1, self.action_space))
        return output

    # # TODO:
    def get_weights(self):
        return self.q_vals

    def set_weights(self, q_vals):
        self.q_vals = q_vals

    # what else do you need?


if __name__ == "__main__":
    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env_kwargs = {
        "height": 3,
        "width": 4,
        "action_dict": action_dict,
        "start_position": (2, 0),
        "reward_position": (0, 3),
    }

    # you can also create your environment like this after installation: env = gym.make('gridworld-v0')
    env = GridWorld(**env_kwargs)

    model_kwargs = {"h": env.height, "w": env.width, "action_space": 4}

    kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 4,
        "total_steps": 100,
        "model_kwargs": model_kwargs
        # and more
    }

    # initilize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    saving_path = os.path.join(os.getcwd(), "progress_gridworld")

    buffer_size = 5000
    test_steps = 1000
    epochs = 20
    sample_size = 1000
    optim_batch_size = 8
    saving_after = 5

    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    manager.initilize_buffer(buffer_size, optim_keys)

    manager.initialize_aggregator(
        path=saving_path,
        saving_after=saving_after,
        aggregator_keys=['loss', 'time_steps']
    )

    print("test before training: ")
    manager.test(
        max_steps=100,
        test_episodes=10,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )

    agent = manager.get_agent()

    for e in range(epochs):

        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)

        sample_dict = manager.sample(sample_size)
        print(f"collected data for: {sample_dict.keys()}")
        data_dict = dict_to_dict_of_datasets(
            sample_dict,
            batch_size=optim_batch_size
        )

        print("optimizing...")

        losses = [
            np.mean(np.random.normal(size=(64, 100)), axis=0) for _ in range(1000)
        ]




    # do the rest!!!!
