import gym
import imageio

env_name = "Pong-v4"
env = gym.make(env_name)
env = gym.wrappers.RecordVideo(env, 'video', episode_trigger=lambda x: x % 2 == 0)

num_episodes = 10

for episode in range(num_episodes):
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        env.render(mode='rgb_array')
