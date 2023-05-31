#import sys
#sys.path.append("/infhome/Documents/mynewproj/venv_1/lib/python3.8/site-packages/")
import common.wrappers
import numpy as np
import wandb
import torch

class Worker(object):
    """The worker is the one which interacts with the env and collects
    rollouts. The worker only collects and does not update the self.models. 
    Here self.env instanace gets created in each thread, which means same envrinment as a copy in different threads.
    """

    def __init__(self, env_name, model_dict, batch_size, gamma, device, id):

        self.device = device

        self.FloatTensor = torch.FloatTensor
        env = common.wrappers.make_atari(env_name, full_action_space=True)
        env = common.wrappers.wrap_deepmind(env, scale=True)
        env = common.wrappers.wrap_pytorch(env)
        
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
        
    def get_batch(self, mode:str="Progress"):
        """Collect rollout for the dataloader

        Returns:
            states, actions, values
        """
        states, actions, rewards, dones = [], [], [], []
        for _ in range(self.batch_size):
            
            # the kb column should watch the active column play
            action = self.model_dict[mode].act(self.state[mode].unsqueeze(0))
            next_state, reward, done, _ = self.env_dict[mode].step(action)
            self.episode_reward[mode] += reward
            
            states.append(self.state[mode])
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            
            if done:
                self.state[mode] = self.FloatTensor(self.env_dict[mode].reset()).to(self.device)
                self.data[mode].append(self.episode_reward[mode])
                
                print(f"Mode {mode}: Worker {self.id} in episode {len(self.data[mode])} Average Score: {np.mean(self.data[mode][-100:])}")
                wandb.log({f"Training Score {mode}-{self.env_dict[mode].spec.id}": np.mean(self.data[mode][-100:])}, commit=False)
                self.episode_reward[mode] = 0
            else:
                self.state[mode] = self.FloatTensor(next_state).to(self.device)
                
        values = self._compute_true_values(states, rewards, dones, mode=mode).unsqueeze(1)
        return states, actions, values, self.data

    
    def _compute_true_values(self, states, rewards, dones, mode):
        """Compute the True values (discounted return) but use value
        estimate of the model for predicition and bootstrapping.

        Args:
            states (_type_): _description_
            rewards (_type_): _description_
            dones (_type_): _description_

        Returns:
            R: discounted return
        """
        R = []
        rewards = self.FloatTensor(rewards).to(self.device)
        dones = self.FloatTensor(dones).to(self.device)
        states = torch.stack(states).to(self.device)
        
        #print(f"rewards {rewards.shape}, states {states.shape}, dones {dones.shape}")
        if dones[-1] == True:
            next_value = rewards[-1]
        else:
            next_value = self.model_dict[mode].get_critic(states[-1].unsqueeze(0))
            
            
        R.append(next_value)
        for i in reversed(range(0, len(rewards) - 1)):
            if not dones[i]:
                next_value = rewards[i] + next_value * self.gamma
            else:
                next_value = rewards[i]
            R.append(next_value)
            
        R.reverse()
        
        return self.FloatTensor(R).to(self.device)