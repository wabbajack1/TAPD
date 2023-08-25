# render the envs
import time 
import gym
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    # list of envs 
    num_envs = 3
    envs = [lambda: gym.make("BreakoutNoFrameskip-v4") for i in range(num_envs)]

    # Vec Env 
    envs = SubprocVecEnv(envs)

    init_obs = envs.reset()

    for i in range(1000):
        actions = [envs.action_space.sample() for i in range(num_envs)]
        envs.step(actions)
        time.sleep(0.001)

    envs.close()