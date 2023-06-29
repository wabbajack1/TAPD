import sys
#sys.path.append("/home/kidimerek/Documents/Studium/Thesis/agnostic_rl-main/lib/python3.9/site-packages/")
import common.wrappers
import gym
import matplotlib.pyplot as plt

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
    env = environment_wrapper("PongNoFrameskip-v4")
    env.seed(0)
    print("The size of frame is: ", env.observation_space.shape)
    print("No. of Actions: ", env.action_space.n)

    # watch an untrained agent
    state = env.reset()[0]
    plt.figure()
    plt.show(block=False)   # Allow updates to the figure window
    sum_rew = 0
    done = False
    while not done:
        m = env.render("rgb_array")
        # plt.imshow(m)
        # plt.draw()
        # plt.pause(0.000000001)    # Add a pause to allow the figure to update
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        sum_rew += reward
        state = next_state
        print(reward)
        if reward < 0:
            print(info)
        if done:
            break 
        
    print(sum_rew)
    plt.imshow(m)
    plt.show()
    env.close()
    pass
