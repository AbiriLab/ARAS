'''

Code for creating and saving n videoclips with random actions for phase 2  

'''
from jaco_env_phase2 import jacoDiverseObjectEnv
import pybullet as p
from itertools import count

number_clips = 5 

env = jacoDiverseObjectEnv(actionRepeat=80, renders=True, isDiscrete=True, maxSteps=30, dv=0.02,
                           removeAutoXDistance=False, width=64, height=64)

for episode in range(number_clips):
    env.reset()

    p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "video/phase2/clips/episode"+str(episode)+".mp4")
    for t in count():
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        if done:
            break
        obs = next_obs
