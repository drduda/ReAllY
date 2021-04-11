# Shared Modular Policy

All relevant scripts are in the ```smp``` package. 

We recommend executing the scripts from the ReAlly root directory using commands like: 
```python smp/smp.py --batch_size 32 --epochs 150 --gamma 0.98 --hidden_units 128 --learning_rate 0.0004 --policy_noise 0.5 --buffer_size 100000 --sample_size 256 --saving_dir "smp_results"```

Note that command line arguments are needed. For further information check the ```-h``` option. 

## Description of the scripts
- ```baseline.py``` trains a monolithic model as a baseline for our expepriment.
- ```smp.py``` trains our experimental model using a shared modular policy approach. 
- ```smp_test.py``` loads a trained model an executes some test episodes to visualize the performance of the model 
  within the environment
- ```base_models.py``` holds some useful base classes for our models. 
- ```smp_utils.py``` holds some helper functions.