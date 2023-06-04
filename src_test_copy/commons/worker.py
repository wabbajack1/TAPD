#import sys
#sys.path.append("/infhome/Documents/mynewproj/venv_1/lib/python3.8/site-packages/")
import sys
sys.path.append("../venv/lib/python3.9/site-packages/")

import common.wrappers
import numpy as np
import wandb
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
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

    def get_batch(self, mode:str="Progress"):
        """Collect rollout for the dataloader

        Returns:
            states, actions, values
        """
        states, actions, rewards, dones = [], [], [], []
        for _ in range(self.batch_size):
            
            # the kb column should watch the active column play
            action = self.model_dict[mode].act(self.state[mode])
            next_state, reward, done  = self.env_dict[mode].step(action)
            self.cumulative_rewards += reward[:,1]
            
            states.append(self.state[mode])
            actions.append(action)
            rewards.append(reward[:,0])
            dones.append(done)
            
            # Check if any of the environments are done
            for i, terminal in enumerate(done):
                if terminal:
                    self.data[mode].append(self.cumulative_rewards[i])
                    print(f"Cumulative reward for environment {i}: {cumulative_rewards[i]}")
                    wandb.log({f"Training Score {mode}-{self.env_dict[mode].spec.id}": np.mean(self.data[mode][-100:])}, commit=False)
                    self.cumulative_rewards[i] = 0  # Reset the cumulative reward for this environment
                else:
                    self.state[mode] = self.FloatTensor(next_state).to(self.device)
                
        values = self._compute_true_values(states, rewards, dones, mode=mode)
        return states, actions, values, self.data

    
    def _compute_true_values(self, states, rewards, dones, mode):
        batch_size = len(states[0])  # Number of environments
        num_steps = len(states)  # Number of steps
        R = torch.zeros((num_steps, batch_size)).to(self.device)  # Initialize R

        # Convert everything to tensors
        rewards = torch.tensor(rewards).float().to(self.device)
        dones = torch.tensor(dones).bool().to(self.device)
        states = torch.stack(states).to(self.device)

        print(self.model_dict[mode].get_critic(states[-1]).shape, dones[-1].shape, rewards[-1].shape)
        # Get bootstrap values
        next_values = torch.where(
            dones[-1],
            rewards[-1],
            self.model_dict[mode].get_critic(states[-1]).squeeze(1)
        )

        R[-1] = next_values
        for i in reversed(range(num_steps - 1)):
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