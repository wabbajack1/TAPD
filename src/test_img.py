import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack
import matplotlib.pyplot as plt
import time
import numpy as np
from envs.wrappers import EpisodicLifeEnv, ClipRewardEnv
from utils import environment_wrapper
import wandb
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import os 

# wandb.init(
#         # set the wandb project where this run will be logged
#         project="atari_single_task",
#         entity="agnostic",
#         monitor_gym=True,
#         # mode="disabled",
#         # id="nd07r8xn",
#         # resume="allow"
#     )


video_dir = './video'
video_path = os.path.join(video_dir, 'video.mp4')

# Ensure the directory exists.
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

env = environment_wrapper("StarGunnerNoFrameskip-v4", mode="human",clip_rewards=False)
# recorder = VideoRecorder(env, path=video_path)

print(env.action_space.n, env.unwrapped.get_action_meanings())

# for _ in range(10):
#     for _ in range(1000):
#         state, reward, done, truncated, info = env.step(env.action_space.sample())
#         recorder.capture_frame()
#     recorder.close()
#     wandb.log({"video": wandb.Video(video_path, fps=4, format="mp4")})


avg_rewards = []
for _ in range(10):
    rewards = 0 
    state = env.reset()

    while True:
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action=action)
        rewards += reward
        if env.was_real_done:
            print(rewards, info)
            break
