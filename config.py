MODEL_NAME = 'FullAuto3obj'
# Declare number of total episodes
num_episodes = 200000      #[25000,100000,500000] 
logfilename = 'learnDQN.log'
# LOGFILE=""
# LOGFILE_POINTER = None
# LOG_ON_SCREEN = False

# Hyperparameters, search for different combinations
BATCH_SIZE = 64 
GAMMA = 0.99

EPS_START = 0.99 # Max of exploration rate
EPS_END = 0.2   # min of exploration rate
EPS_DECAY_LAST_FRAME = 10e4 # last episode to have decay in exploration rate

TARGET_UPDATE = 1000
LEARNING_RATE = 1e-3
REPLAY_BUFFER_SIZE = 30000
Stack_Size = 4