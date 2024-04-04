########## Test ############
modelPath = "./models/pickplace_bs64_ss4_rb30000_gamma0.99_decaylf100000.0_lr0.001.pt"



########### TRAIN ##########

MODEL_NAME = 'pickplace'

# Declare number of total episodes
num_episodes = 200000      #[25000,100000,500000] 
logfilename = 'learnDQN.log'

# Hyperparameters, search for different combinations
BATCH_SIZE = 32 
GAMMA = 0.99

EPS_START = 0.99 # Max of exploration rate
EPS_END = 0.2   # min of exploration rate
EPS_DECAY_LAST_FRAME = 10e4 # last episode to have decay in exploration rate

TARGET_UPDATE = 1000
LEARNING_RATE = 1e-3
REPLAY_BUFFER_SIZE = 30000
Stack_Size = 4