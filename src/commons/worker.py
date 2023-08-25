import envs.wrappers
import numpy as np
import wandb
import torch
from typing import Optional
from copy import deepcopy
from gym.wrappers import AtariPreprocessing, FrameStack
from envs.wrappers import ClipRewardEnv
import gym
from utils import environment_wrapper

# keep track of the number of frames
frame_number = {}
episode_number = {}
global_steps = {}

class Worker(object):
    """The worker is the one which interacts with the env and collects
    rollouts. The worker only collects and does not update the self.models. 
    Here self.env instanace gets created in each thread, which means same envrinment as a copy in different threads.
    """

    def __init__(self, env_name, batch_size, gamma, device, seed, rank):        
        self.device = device
        self.env_name = env_name
        self.rank = rank

        self.FloatTensor = torch.FloatTensor
        env = environment_wrapper(env_name=env_name, mode="rgb_array", clip_rewards=True)
        env.seed(seed+rank)
        print(f"Rank: {self.rank} - env: {id(env)}\n")
        
        self.env_dict = {}
        self.env_dict["Progress"] = env
        self.env_dict["Compress"] = env
        
        self.state = {}
        self.state["Progress"] = self.FloatTensor(np.array(self.env_dict["Progress"].reset()))
        self.state["Compress"] = self.FloatTensor(np.array(self.env_dict["Compress"].reset()))
        
        self.batch_size = batch_size
        self.gamma = gamma


        # store data for each task
        self.data = {}
        self.data["Progress"] = []
        self.data["Compress"] = []
        
        self.episode_reward = {}
        self.episode_reward["Progress"] = 0
        self.episode_reward["Compress"] = 0
        self.episode_reward_orginal = 0
        
    def get_batch(self, mode:Optional[str]="Progress", batch_size:Optional[int]=None, model_dict=None, nmb_of_workers:int=None):
        """Create rollout for update of parameters. This method is called after each training step. Its is strongly
        dependent on the memory/memory.py module, which stores the experience for further batch creation via dataloader
        during training.
        
        Args:
            mode (Optional[str], optional): Switch between models,. Defaults to "Progress"
        Returns:
            tuple: (states, actions, values, info["frame_number"]). info["frame_number"] is optional
        """
        self.model_dict = model_dict
        global frame_number
        global global_steps
        
        states, actions, rewards, dones = [], [], [], []
        
        # treat batchsize differently during modes (in progress mode == rl setting, in compress+ewc mode == more supervised learning)
        self.batch_size_mode = self.batch_size if mode == "Progress" else batch_size

        for collect in range(self.batch_size_mode):
            action = self.model_dict[mode].act(self.state[mode].unsqueeze(0).to(self.device))
            next_state, reward, done, _, info = self.env_dict[mode].step(action)
            self.episode_reward[mode] += reward[1]
            states.append(self.state[mode]) 
            actions.append(action)
            rewards.append(reward[0])
            dones.append(done)
            # print(self.state[mode].unsqueeze(0).sum(), self.rank, action, info)
            
    
            if self.rank == 0 and mode=="Progress":
                global_steps.setdefault(self.env_name, 0)
                global_steps[self.env_name] += nmb_of_workers
                    
            if done:
                frame_number.setdefault(mode, {}).setdefault(self.env_name, 0)
                episode_number.setdefault(mode, {}).setdefault(self.env_name, 0)
                
                if self.env_dict[mode].game_over:
                    self.data[mode].append(self.episode_reward[mode])
                    print(f"Mode {mode} -- Worker {self.rank} in episode {episode_number[mode][self.env_name]} -- Average Score: {np.mean(self.data[mode][-100:])} -- Total frames across threads: {frame_number[mode][self.env_name]}")
                    if self.rank == 0:
                        frame_number[mode][self.env_name] += info['episode_frame_number'] # each info is accociated with one thread
                        episode_number[mode][self.env_name] += 1
                        wandb.log({f"Training/Training Score {mode}-{self.env_dict[mode].spec.id}": np.mean(self.data[mode][-100:]), 
                                   f"Training/Training Frame-# {self.env_dict[mode].spec.id}": frame_number[mode][self.env_name],
                                   f"Training/Training Steps-# {self.env_dict[mode].spec.id}": global_steps[self.env_name]})
                        
                    self.episode_reward[mode] = 0
                
                self.state[mode] = self.FloatTensor(np.array(self.env_dict[mode].reset())) # no-op step to advance from terminal/lost life state, when lives > 0
            else:
                self.state[mode] = self.FloatTensor(np.array(next_state))
            
            # print(f"Process end {collect}, Rank {self.rank}, frame {info}")    
        values = self._compute_true_values(states, rewards, dones, mode=mode)
        # print(values.shape, self.rank)
        return states, actions, values
 
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
        rewards = self.FloatTensor(rewards)
        dones = self.FloatTensor(dones)
        states = torch.stack(states)
        
        #print(states[-1].shape, states.shape, dones.shape, rewards.shape)
        #print(f"rewards {rewards.shape}, states {states.shape}, dones {dones.shape}")
        if dones[-1] == True:
            next_value = rewards[-1]
        else:
            next_value = self.model_dict[mode].get_critic(states[-1].unsqueeze(0).to(self.device)).squeeze(1)
        
        R.append(next_value)
        for i in reversed(range(0, len(rewards) - 1)):
            if not dones[i]:
                next_value = rewards[i] + next_value * self.gamma
            else:
                next_value = rewards[i]
            R.append(next_value)
            
        R.reverse()
        
        return self.FloatTensor(R).unsqueeze(1)