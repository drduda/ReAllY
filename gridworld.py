import gym
import numpy as np
import ray
from really import SampleManager
from gridworlds import GridWorld
import os


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
        # todo proper Initialization?
        self.q_values = np.random.random((h, w, action_space))

    def __call__(self, state):
        ## # TODO:
        h = int(state[0][0])
        w = int(state[0][1])
        output = {}
        output["q_values"] = [self.q_values[h, w, :]]
        return output

    # # TODO:
    def get_weights(self):
        return None

    def set_weights(self, q_vals):
        pass

    # what else do you need?


if __name__ == "__main__":
    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env_kwargs = {
        "height": 10,
        "width": 10,
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
        "num_parallel": 2,
        "total_steps": 100,
        "model_kwargs": model_kwargs,
        "env_kwargs" : env_kwargs
        # and more
    }

    # initilize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    buffer_size = 500
    test_steps = 50
    epochs = 20
    sample_size = 100
    optim_batch_size = 8
    saving_after = 5

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    #saving_path = os.getcwd() + "/progress_test"
    #manager.initialize_aggregator(
    #    path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    #)

    # initial testing:
    #print("test before training: ")
    #manager.test(test_steps, do_print=True)

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):
        print("collecting experience..")

        # Gives you state action reward trajetories
        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)

