# One Policy to Control Them All
![blub](https://github.com/drduda/ReAllY/blob/master/images/salto.gif)
We implemented the paper [One Policy to Control Them All](https://arxiv.org/pdf/2007.04976v1.pdf) with the OpenAI gym environment [**BipedalWalker**](https://gym.openai.com/envs/BipedalWalker-v2/). The main idea of the paper is to use the **same** *Shared Modular Policy (SMP)* for all actuators of an kinetic agent and let the actuators **communicate** with each other in order to enable **global coordination**. We are the first, who use a *SMP* for the Bidpeal Walker environment. 
For a quick overview about our results, have a look at [this video](https://github.com/drduda/ReAllY/blob/master/show_performance.mp4). 
For a detailed graphical conceptual overview about the paper as well as a scientific comparision between the baseline and shared modular, have a look at our [report](https://github.com/drduda/ReAllY/blob/master/DRL_project__One_policy_to_control_them_all.pdf). 

## Run

All relevant scripts are in the ```smp``` package. 

We recommend executing the scripts from the ReAlly root directory using commands like: 
```python smp/smp.py --batch_size 32 --epochs 150 --gamma 0.98 --hidden_units 128 --learning_rate 0.0004 --policy_noise 0.5 --buffer_size 100000 --sample_size 256 --saving_dir "smp_results"```

Note that command line arguments are needed. For further information check the ```-h``` option. 

## Scripts
- ```baseline.py``` trains a monolithic model as a baseline for our expepriment.
- ```smp.py``` trains our experimental model using a shared modular policy approach. 
- ```smp_test.py``` loads a trained model an executes some test episodes to visualize the performance of the model 
  within the environment
- ```base_models.py``` holds some useful base classes for our models. 
- ```smp_utils.py``` holds some helper functions.

## Result Folders (chronologically following creation date)
Each run was based on configuration 8 from our best grid search result which can be found in 
```grid_search/results_1.md```.

1. ```grid_search/results_1.md```: summarized results of our grid search 
2. ```baseline_results```: Monolithic baseline model
3. ```smp_results_config8```: simple SMP run with best configuration from grid search
4. ```smp_results_config8_bigger_buffer``` increased buffer size to 100,000
5. ```smp_results_moreepochs``` increased number of epochs to 150
6. ```smp_results_config8_withlidar```: added lidar to states

Authors: Niklas Dettmer & Marko Duda
