import gym
from stable_baselines3 import A2C
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

import common.wrappers

# Create a function to make Atari environments
def make_atari_env(env_name):
    def _init_env():
        env = common.wrappers.make_atari(env_name, full_action_space=True)
        env = common.wrappers.wrap_deepmind(env, scale=True, clip_rewards=True)
        env = common.wrappers.wrap_pytorch(env)
        return env
    return _init_env

def main():
    env = SubprocVecEnv([make_atari_env('PongNoFrameskip-v4') for _ in range(4)])
    print(env.reset())
    # Create the A2C agent
    #model = A2C('CnnPolicy', env, verbose=1)

    # Train the agent
    #model.learn(total_timesteps=100000)

    # Save the trained agent
    #model.save("a2c_pong")

    # Load the trained agent
    #loaded_model = A2C.load("a2c_pong")

    # Evaluate the agent
    #mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=10)

    # Print the evaluation results
    #print(f"Mean reward: {mean_reward} +/- {std_reward}")

if __name__ == '__main__':
    main()
