# RL4AD - Reinforcement Learning for Anomaly Detection

This needs to be used in combination with gym-network_intrusion library

# Installation
# Step 1: 
Install all the required python packages described in requirements.txt

# Step 2: 
Install gym from OpenAI using the following steps
1. git clone https://github.com/openai/gym.git
2. cd gym
3. pip install -e .

# Step 3: 
Install gym-network_intrusion using the following steps
1. git clone https://github.com/harik68/gym-network_intrusion.git
2. cd gym-network_intrusion
3. pip install -e .

# Step 4: 
Install RL4AD using the following steps
1. git clone https://github.com/harik68/RL4AD.git
2. Copy the following file into the directory datasets
   https://www.dropbox.com/s/e2n5ow6b117ub80/kdd_nsl_train_onehot_string.pkl?dl=0
3. In your code create an instance of gym_network_intrusion environment using the following commands
4. Run the code NSL_KDD_DNN_QLearning.py

Remember to clear tmp folder after each run
