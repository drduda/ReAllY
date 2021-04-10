from really import SampleManager
from copy import copy
import os
import gym
from smp import TD3Net
import ray

if __name__ == "__main__":
    hidden_units = 64
    msg_dim = 16
    model_path = os.getcwd() + "/smp_results_config8_bigger_buffer"

    ray.init(log_to_driver=False)

    env_test_instance = gym.make('BipedalWalker-v3')
    model_kwargs = {
        # action dimension for modular actions
        'action_dimension': 1,
        'min_action': copy(env_test_instance.action_space.low)[0],
        'max_action': copy(env_test_instance.action_space.high)[0],
        'msg_dimension': msg_dim,
        'fix_sigma': True,
        'hidden_units': hidden_units
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

    manager.load_model(model_path)
    manager.test(200, test_episodes=5, render=True)

    ray.shutdown()
