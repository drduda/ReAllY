import gym
from gym.envs.box2d.bipedal_walker import BipedalWalker


class MyBipedalWalker(BipedalWalker):

    def __init__(self):
        super().__init__()
        # TODO: override self.action_space and self.observation_space

    def step(self, action):
        # TODO: work with new action_space and observation_space
        # TODO: remember to not execute anything for the head joint
        super().step(action)
