from gym.envs.registration import register
from gridworlds.envs.gridworld import GridWorld
from gridworlds.envs.gridworld_global import GridWorld_Global

register(
    id="mybipedalwalker-v0",
    entry_point="bipedalwalker.envs:MyBipedalWalker",
    max_episode_steps=100000,
)
