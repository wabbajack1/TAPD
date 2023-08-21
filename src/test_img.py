import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack
import matplotlib.pyplot as plt
import time
import numpy as np
from envs.wrappers import EpisodicLifeEnv, ClipRewardEnv

env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
env = AtariPreprocessing(env=env, scale_obs=True, terminal_on_life_loss=False)
# env = EpisodicLifeEnv(env=env)
env = ClipRewardEnv(env=env)
env = FrameStack(env=env, num_stack=4)

state, info = env.reset()
print(env.action_space.n, env.get_action_meanings())
state, reward, done, truncated, info = env.step(2)
state, reward, done, truncated, info = env.step(2)
state, reward, done, truncated, info = env.step(2)

for img in np.array(state):
    print(info)
    img_imshow = plt.imshow(img)
    plt.colorbar(img_imshow)
    plt.show()

state, reward, done, truncated, info = env.step(3)

for img in np.array(state):
    print(info)
    img_imshow = plt.imshow(img)
    plt.colorbar(img_imshow)
    plt.show()
