import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################
########### TESTING PARAMETERS ##########
RENDER = False
modelPath = "./models/ARAS_v7_bs64_ss4_rb30000_gamma0.5_decaylf20000_lr1e-05.pt" # ARAS
# modelPath = "./models/DQN_baseline_v2_bs64_ss4_rb30000_gamma0.3_decaylf5000_lr1e-05.pt" # DQN
SCENARIO = "dynamic_both" # ["fixed", "dynamic_pickup", "dynamic_dropoff", "dynamic_both"]
EPISODE_NUMBER = 50

##########################################
########### TRAINING PARAMETERS ##########
PRETRAINED_MODEL_PATH = ""
MODEL_NAME = 'ARAS' # ['DQN', 'ARAS']

num_episodes = 20000  #[25000,100000,500000] 
logfilename = 'learnDQN.log'

# Hyperparameters
BATCH_SIZE = 64 
GAMMA = 0.8
EPS_START = 0.8 # Max of exploration rate
EPS_END = 0.1  # min of exploration rate
EPS_DECAY_LAST_FRAME = 5000 # last episode to have decay in exploration rate
TARGET_UPDATE = 1000
LEARNING_RATE = 1e-5
REPLAY_BUFFER_SIZE = 30000
Stack_Size = 4
