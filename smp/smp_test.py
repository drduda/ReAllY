from really import SampleManager
from copy import copy
import os
import gym
import ray
import sys
import argparse


def parse(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--msg_dim", default=1, type=int)
    parser.add_argument("--hidden_units", default=128, type=int)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--baseline", default=False, type=bool)

    return parser.parse_args()


def main(args):
    hidden_units = args.hidden_units
    msg_dim = args.msg_dim
    model_path = os.getcwd() + "/" + args.model_dir

    ray.init(log_to_driver=False)

    env_test_instance = gym.make('BipedalWalker-v3')

    if args.baseline:
        from baseline import TD3Net
        action_dimension = copy(env_test_instance.action_space.shape)
    else:
        from smp import TD3Net
        action_dimension = 1

    model_kwargs = {
        # action dimension for modular actions
        'action_dimension': action_dimension,
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


if __name__ == "__main__":
    args = parse(sys.argv[1:])
    main(args)
