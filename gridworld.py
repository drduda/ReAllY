import gym
import numpy as np
import ray
from really import SampleManager
from gridworlds import GridWorld
import os
from datetime import datetime


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
        state = state.astype(int)
        output = {}
        output["q_values"] = self.q_values[state[:, 0], state[:, 1], :]
        return output

    def get_weights(self):
        return self.q_values

    def set_weights(self, q_vals):
        self.q_values = q_vals

    # what else do you need?
    def save(self, filepath, **kwargs):
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        filename = datetime.now().strftime("%Y-%m-%d_%H-%M") + '_gridworld_weights.npy'

        with open(os.path.join(filepath, filename), 'wb') as f:
            np.save(f, self.q_values)

    def save_weights(self, filepath, **kwargs):
        self.save(filepath, **kwargs)

    def load_weights(self, filepath, **kwargs):
        with open(filepath, 'rb') as f:
            self.q_values = np.load(f)
        self.action_space = self.q_values.shape[-1]


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
        "num_parallel": 2,
        "total_steps": 100,
        "model_kwargs": model_kwargs
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
    test_episodes = 5

    learning_rate = 0.1
    discount = .98


    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    saving_path = os.getcwd() + "/progress_gridworld"
    manager.initialize_aggregator(
        path=saving_path, saving_after=saving_after, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:
    print("test before training: ")
    manager.test(test_steps, do_print=True)

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):
        print("collecting experience..")

        # Gives you state action reward trajetories
        data = manager.get_data()
        manager.store_in_buffer(data)

        # Sample experience
        sample_dict = manager.sample(sample_size)
        state_t = sample_dict['state']
        action_t = sample_dict['action']

        q_values_t = agent.q_val(state_t, action_t)

        #todo not_done

        # Get q_values of t=1
        q_values_t_1 = agent.max_q(sample_dict['state_new'])

        # Update q values
        delta = sample_dict['reward'] + discount * q_values_t_1 - q_values_t
        q_values_updated = q_values_t + learning_rate * delta

        # Put updated q values into agent
        new_weights = agent.get_weights().copy()
        new_weights[state_t[:, 0], state_t[:, 1], action_t] = q_values_updated
        agent.set_weights(new_weights)

        manager.set_agent(new_weights)
        agent = manager.get_agent()

        time_steps = manager.test(test_steps)

        manager.update_aggregator(loss=delta, time_steps=time_steps)

        print(
            f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in delta])}   avg env steps ::: {np.mean(time_steps)}"
        )

        if e % saving_after == 0:
            manager.save_model(saving_path, e)

    #manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=test_episodes, render=True)

