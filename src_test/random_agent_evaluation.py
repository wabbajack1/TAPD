import sys
#sys.path.append("/home/kidimerek/Documents/Studium/Thesis/agnostic_rl-main/lib/python3.9/site-packages/")
import common.wrappers
import gym
import matplotlib.pyplot as plt
import numpy as np
import time

def environment_wrapper(env_name):
    """Preprocesses the environment based on the wrappers

    Args:
        env_name (string): env name

    Returns:
        env object: return the preprocessed env (MDP problem)
    """
    env = common.wrappers.make_atari(env_name)
    env = common.wrappers.wrap_deepmind(env, scale=True, clip_rewards=True)
    env = common.wrappers.wrap_pytorch(env)

    return env


if __name__ == "__main__":
    env = environment_wrapper("SpaceInvadersNoFrameskip-v4")
    env.seed(0)
    print("The size of frame is: ", env.observation_space.shape)
    print("No. of Actions: ", env.action_space.n)

    # watch an untrained agent
    sum_rew = 0
    avg_rew = []
    done = False
    episodes = 100
    frames = 0
    # Start the timer
    start_time = time.time()
    
    for _ in range(episodes):
        state = env.reset()[0]
        while not env.was_real_done:
            m = env.render("rgb_array")
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            sum_rew += reward[1]
            state = next_state
            
            if env.was_real_done:
                avg_rew.append(sum_rew)
                sum_rew = 0
                break
            
    elapsed_time = time.time() - start_time
    #plt.imshow(m)
    #plt.show()
    env.close()
    print(f"list of rewards {avg_rew}, avg reward over #-epsiodes {np.mean(avg_rew)}, elapsed_time {elapsed_time}")
    pass
