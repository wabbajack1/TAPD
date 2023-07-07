#import sys
#sys.path.append("/infhome/Documents/mynewproj/venv_1/lib/python3.8/site-packages/")
import sys
sys.path.append("../venv/lib/python3.9/site-packages/")

import common.wrappers
import numpy as np
import wandb
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from commons.model import KB_Module, Active_Module, ProgressiveNet


class Worker(object):
    """The worker is the one which interacts with the env and collects
    rollouts. The worker only collects and does not update the self.models. 
    Here self.env instanace gets created in each thread, which means same envrinment as a copy in different threads.
    """

    def __init__(self, env_name, model_dict, batch_size, gamma, device, num_envs):

        self.device = device
        self.FloatTensor = torch.FloatTensor
        env = SubprocVecEnv([self.make_env(env_name) for _ in range(num_envs)])
        self.env_name = env_name

        
        self.env_dict = {}
        self.env_dict["Progress"] = env
        self.env_dict["Compress"] = env
        
        self.state = {}
        self.state["Progress"] = self.FloatTensor(self.env_dict["Progress"].reset()).to(self.device)
        self.state["Compress"] = self.FloatTensor(self.env_dict["Compress"].reset()).to(self.device)
        
        self.model_dict = model_dict
        self.batch_size = batch_size
        self.gamma = gamma
        self.id = id


        # store data for each task
        self.data = {}
        self.data["Progress"] = []
        self.data["Compress"] = []
        
        self.episode_reward = {}
        self.episode_reward["Progress"] = 0
        self.episode_reward["Compress"] = 0
        self.cumulative_rewards = np.zeros(num_envs)
    
    def make_env(self, env_name):
        def _init():
            env = common.wrappers.make_atari(env_name, full_action_space=True)
            env = common.wrappers.wrap_deepmind(env, scale=True)
            env = common.wrappers.wrap_pytorch(env)
            return env
        return _init


    # def get_batch(self, mode:str="Progress"):
    #     """
    #     Get a batch of experiences from the vectorized environment.

    #     Args:

    #     Returns:
    #         A batch of experiences and the computed true and bootstrap values.
    #     """

    #     states = []
    #     actions = []
    #     rewards = []
    #     next_states = []
    #     dones = []
    #     true_values = []
    #     bootstrap_values = []

    #     for _ in range(self.batch_size):
    #         action = self.model_dict[mode].act(self.state[mode]) # Assuming the agent has a method to select actions
    #         next_state, reward, done, info = self.env_dict[mode].step(action)
    #         self.cumulative_rewards += reward
            
    #         for i in range(self.env_dict[mode].num_envs):
    #             # If done, get the terminal observation from the info dict
    #             if done[i]:
    #                 terminal_observation = info[i].get('terminal_observation')
    #                 if terminal_observation is not None:
    #                     next_state[i] = self.FloatTensor(terminal_observation).to(self.device)
                        
    #                 # logging
    #                 self.data[mode].append(self.cumulative_rewards[i])
    #                 print(f"Cumulative reward for environment {self.env_name}-{i}: {self.cumulative_rewards[i]}; Episodes: {len(self.data[mode])}")
    #                 wandb.log({f"Training Score {mode}-{self.env_name}": np.mean(self.data[mode][-100:]), "Frame-#":info[i]["frame_number"]}, commit=False)
    #                 self.cumulative_rewards[i] = 0  # Reset the cumulative reward for this environment

    #             # Compute the true value (discounted future reward)
    #             if done[i]:
    #                 true_value = reward[i]
    #             else:
    #                 true_value = reward[i] + gamma * self.agent.estimate_value(next_state[i])

    #             # Compute the bootstrap value
    #             bootstrap_value = reward[i] + gamma * self.agent.estimate_value(next_state[i]) * (1 - done[i])

    #             # Store the experience
    #             states.append(self.state[mode][i])
    #             actions.append(action[i])
    #             rewards.append(reward[i])
    #             next_states.append(next_state[i])
    #             dones.append(done[i])
    #             true_values.append(true_value)
    #             bootstrap_values.append(bootstrap_value)

    #         # Update the current state
    #         self.state[mode] = self.FloatTensor(next_state).to(self.device)
    #     return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), np.array(true_values), np.array(bootstrap_values)


    def get_batch(self, mode:str="Progress"):
        states, actions, rewards, dones = [], [], [], []
        for _ in range(self.batch_size):
            for experience in self.state[mode]:
                print(experience.shape)
                action = self.model_dict[mode].act(self.state[mode])
                next_state, reward, done, info = self.env_dict[mode].step(action)
                self.cumulative_rewards += reward[:,1]
                
                states.append(self.state[mode])
                actions.append(action)
                rewards.append(reward[:,0])
                dones.append(done)

            # Check if any of the environments are done
            for i, terminal in enumerate(done):
                if terminal:
                    #final_state = info[i]['terminal_observation']  # Use terminal_observation as the final state
                    self.data[mode].append(self.cumulative_rewards[i])
                    avg_data = np.mean(self.data[mode][-100:])
                    print(f"Cumulative reward for environment {self.env_name}-{i}: {avg_data}; Episodes: {len(self.data[mode])}")
                    wandb.log({f"Training Score {mode}-{self.env_name}": avg_data, "Frame-#":info[i]["frame_number"]}, commit=False)
                    self.cumulative_rewards[i] = 0  # Reset the cumulative reward for this environment
            
            # get next obs, init observation is set automatically
            self.state[mode] = self.FloatTensor(next_state).to(self.device)
            
        states = torch.stack(states).permute(0, 1, 2, 3, 4)
        actions = torch.stack(actions).permute(0, 1)
        dones = torch.from_numpy(np.stack(dones)).permute(0, 1)
        rewards = torch.from_numpy(np.stack(rewards)).permute(0, 1)

        values = self._compute_true_values(states, rewards, dones, mode=mode)
        return states, actions, values, self.data

    
    def _compute_true_values(self, states, rewards, dones, mode):
        env_size = len(states[0])  # Number of environments == #-workers
        batch_size = len(states)  # batch_size == Number of steps in the env
        R = torch.zeros((batch_size, env_size)).to(self.device)  # Initialize R
        
        # Convert everything to tensors
        rewards = rewards.float().to(self.device)
        dones = dones.bool().to(self.device)
        states = states.to(self.device)
        
        # Get bootstrap values
        next_values = torch.where(
            dones[-1],
            rewards[-1],
            self.model_dict[mode].get_critic(states[-1]).squeeze(1)
        )
        
        R[-1] = next_values
        for i in reversed(range(batch_size - 1)):
            R[i] = torch.where(
                dones[i],
                rewards[i],
                rewards[i] + self.gamma * next_values
            )
            next_values = R[i]

        return R

if __name__ == "__main__":
    active_model = Active_Module("cuda:0", lateral_connections=False).to("cuda:0")   
    kb_model = KB_Module("cuda:0").to("cuda:0")
    progNet = ProgressiveNet(kb_model, active_model).to("cuda:0") 
    worker = Worker("PongNoFrameskip-v4", {"Progress":progNet}, 20, 0.99, "cuda:0", 4)

    states, actions, values, _ = worker.get_batch()

    print(values.shape)