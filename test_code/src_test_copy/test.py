import sys
sys.path.append("/home/kidimerek/Documents/Studium/Thesis/agnostic_rl-main/lib/python3.9/site-packages/")
import common.wrappers

def environment_wrapper(env_name):
    """Preprocesses the environment based on the wrappers

    Args:
        env_name (string): env name

    Returns:
        env object: return the preprocessed env (MDP problem)
    """
    env = common.wrappers.make_atari(env_name)
    env = common.wrappers.wrap_deepmind(env, scale=True)
    env = common.wrappers.wrap_pytorch(env)

    return env


if __name__ == "__main__":
    env = environment_wrapper('PongNoFrameskip-v4')
    print(env.sample())
    pass