import sys
sys.path.append("/home/kidimerek/Documents/Studium/Thesis/agnostic_rl-main/lib/python3.9/site-packages/")

import common.wrappers
import numpy as np
import wandb
import torch

class Worker(object):
    """The worker is the one which interacts with the env and collects
    rollouts. The worker only collects and does not update the self.models.
    """

    def __init__(self, env_name, model, batch_size, gamma):

        if torch.cuda.is_available():
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

        self.env = common.wrappers.make_atari(env_name)
        self.env = common.wrappers.wrap_deepmind(self.env, scale=True)
        self.env = common.wrappers.wrap_pytorch(self.env)

        self.episode_reward = 0
        self.state = self.FloatTensor(self.env.reset())
        self.model = model
        self.batch_size = batch_size
        self.gamma = gamma


        self.data = {
            'episode_rewards': []
        }
        
    def get_batch(self):
        """Collect rollout. This method can also be used for evaluation after training.

        Returns:
            states, actions, values
        """
        states, actions, rewards, dones = [], [], [], []
        for _ in range(self.batch_size):
            action = self.model.act(self.state.unsqueeze(0))
            next_state, reward, done, _ = self.env.step(action)
            self.episode_reward += reward

            states.append(self.state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            if done:
                self.state = self.FloatTensor(self.env.reset())
                self.data['episode_rewards'].append(self.episode_reward)
                print(f"Average Score: {np.mean(self.data['episode_rewards'][-100:])}")
                #wandb.log({"Score": np.mean(self.data['episode_rewards'][-100:])}, commit=False)
                self.episode_reward = 0
            else:
                self.state = self.FloatTensor(next_state)
                
        values = self._compute_true_values(states, rewards, dones).unsqueeze(1)
        return states, actions, values

    
    def _compute_true_values(self, states, rewards, dones):
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
        
        if dones[-1] == True:
            next_value = rewards[-1]
        else:
            next_value = self.model.get_critic(states[-1].unsqueeze(0))
            
        R.append(next_value)
        for i in reversed(range(0, len(rewards) - 1)):
            if not dones[i]:
                next_value = rewards[i] + next_value * self.gamma
            else:
                next_value = rewards[i]
            R.append(next_value)
            
        R.reverse()
        
        return self.FloatTensor(R)