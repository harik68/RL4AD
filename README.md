# RL4AD - Reinforcement Learning for Anomaly Detection

This gym environment simulates network intrusion similar to that described in the NSL-KDD Dataset described in this link http://www.unb.ca/cic/datasets/nsl.html

# installation
First install gym from OpenAI using the following steps
1. git clone https://github.com/openai/gym.git
2. cd gym
3. pip install -e .

After this install gym-network_intrusion using the following staps
1. git clone https://github.com/harik68/gym-network_intrusion.git
2. cd gym-network_intrusion
3. pip install -e .

# usage in your program
1. First create a directory named datasets in your folder containing the main programme
2. Copy the following file into this directory
   https://www.dropbox.com/s/e2n5ow6b117ub80/kdd_nsl_train_onehot_string.pkl?dl=0
3. In your code create an instance of gym_network_intrusion environment using the following commands
import gym
import gym_network_intrusion
env = gym.make('network-intrusion-v0')

After this you can use the Jupyter Notebook NSL_KDD_DNN_QLearning.ipynb which contains the main programe for Anomaly Detection

