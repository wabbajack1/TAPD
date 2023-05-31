import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from runner import environment_wrapper

def make_env():
    def _init():
        env = environment_wrapper(None, "PongNoFrameskip-v4", False)
        return env
    return _init

if __name__ == '__main__':
    env = SubprocVecEnv([make_env() for _ in range(4)])
    print(env.reset())