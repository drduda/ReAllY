# Shared Modular Policy

All relevant scripts are in the ```smp``` package. Note that command line arguments are needed. For further information 
check the ```-h``` option. 
- ```baseline.py``` trains a monolithic model as a baseline for our expepriment.
- ```smp.py``` trains our experimental model using a shared modular policy approach. 
- ```smp_test.py``` loads a trained model an executes some test episodes to visualize the performance of the model 
  within the environment
- ```base_models.py``` holds some useful base classes for our models. 
- ```smp_utils.py``` holds some helper functions.