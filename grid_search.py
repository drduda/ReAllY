from contextlib import redirect_stdout
import argparse
import os

# Module to test grid search
from smp import main

PATH = './grid_search/'
if not os.path.isdir(PATH):
    os.mkdir(PATH)

# Hyperparams
EPOCHS = [10]
BATCH_SIZE = [32]
POLICY_NOISE = [0.13, 0.5]
MSG_DIM = [16, 32]
LEARNING_RATE = [4e-4, 4e-3]
HIDDEN_UNITS = [100, 300]
GAMMA = [0.98, 0.99]

id = 0

for epochs in EPOCHS:
    for batch_size in BATCH_SIZE:
        for policy_noise in POLICY_NOISE:
            for msg_dim in MSG_DIM:
                for learning_rate in LEARNING_RATE:
                    for hidden_units in HIDDEN_UNITS:
                        for gamma in GAMMA:

                            # Add arguments
                            args = argparse.ArgumentParser().parse_args()
                            args.epochs = epochs
                            args.batch_size = batch_size
                            args.policy_noise = policy_noise
                            args.msg_dim = msg_dim
                            args.learning_rate = learning_rate
                            args.hidden_units = hidden_units
                            args.gamma = gamma

                            print("Job running: "+str(id))

                            # Save prints to log file
                            with open(PATH+str(id)+'.log', 'w') as f:
                                with redirect_stdout(f):
                                    main(args)

                            id += 1

