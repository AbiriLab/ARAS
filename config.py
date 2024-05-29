########## Test ############
modelPath = "./models/pickplace_seg_v4_bs64_ss4_rb30000_gamma0.99_decaylf120000_lr0.001.pt"
# modelPath = "/home/tnlab/Projects/github/sharedcontrol_DQN_Kinova/models/pickplace_seg_v4_bs64_ss4_rb30000_gamma0.99_decaylf120000_lr0.001.pt"


########### TRAIN ##########
PRETRAINED_MODEL_PATH = "./models/pickplace_seg_bs64_ss4_rb30000_gamma0.99_decaylf40000.0_lr0.001.pt"
# PRETRAINED_MODEL_PATH = ""

MODEL_NAME = 'pickplace_seg_v4'

# Declare number of total episodes
num_episodes = 500000      #[25000,100000,500000] 
logfilename = 'learnDQN.log'

# Hyperparameters, search for different combinations
BATCH_SIZE = 64 
GAMMA = 0.99

EPS_START = 0.99 # Max of exploration rate
EPS_END = 0.3  # min of exploration rate
EPS_DECAY_LAST_FRAME = 120000 # last episode to have decay in exploration rate

TARGET_UPDATE = 1000
LEARNING_RATE = 1e-3
REPLAY_BUFFER_SIZE = 30000
Stack_Size = 4