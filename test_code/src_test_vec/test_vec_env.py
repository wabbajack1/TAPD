
from stable_baselines3 import A2C
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

import envs.wrappers

# Create a function to make Atari environments
def make_atari_env(env_name):
    def _init_env():
        env = envs.wrappers.make_atari(env_name, full_action_space=True)
        env = envs.wrappers.wrap_deepmind(env, scale=True, clip_rewards=True)
        env = envs.wrappers.wrap_pytorch(env)
        return env
    return _init_env

def main():
    env = SubprocVecEnv([make_atari_env('PongNoFrameskip-v4') for _ in range(4)])
    print(env.reset())

if __name__ == '__main__':
    main()
