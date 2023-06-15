from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from typing import Optional
from commons.memory.CustomDataset import CustomDataset
from torch.utils.data.dataloader import DataLoader

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, agent:None, model: nn.Module, ewc_gamma=0.4, device=None, env_name:Optional[str] = None):
        """The online ewc algo 
        Args:
            task (None): the task (in atari a env) for calculating the importance of task w.r.t the paramters
            model (nn.Module): the model which params are important to protect
            ewc_gamma (float, optional): the deacay factor. Defaults to 0.4.
            device (_type_, optional): _description_. Defaults to None.
        """
        if torch.cuda.is_available():
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

        self.model = model
        self.device = device
        self.ewc_gamma = ewc_gamma
        self.env_name = env_name
        self.agent = agent # we need the memory module of this object (in atari domain task == env == data)

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.mean_params = {}
        self.old_fisher = None
        self.fisher = self.calculate_fisher() # calculate the importance of params for the previous task
        
        for n, p in deepcopy(self.params).items():
            self.mean_params[n] = variable(p.data)
            
    def calculate_fisher(self):
        print(f"Calculation of the task for the importance of each parameter: {self.env_name}")
        
        self.model.eval()
        fisher = {}
        
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            fisher[n] = variable(p.data)
        
        states, actions, true_values = self.agent.memory.pop_all()
        self.agent.memory.delete_memory()
        dataset = CustomDataset(states, actions, true_values)
        dataloader = DataLoader(dataset, batch_size=self.agent.batch_size, shuffle=False)
        
        state = self.FloatTensor(self.task.reset()).to(self.device)
        for batch_states, batch_actions, batch_true_values in dataloader:
            # Calculate gradients
            self.model.zero_grad()
            
            batch_states = batch_states.to(self.device)
            batch_actions = batch_actions.to(self.device)
            
            
            action = self.model.act(state.unsqueeze(0).to(self.device))
            next_state, reward, done, _ = self.task.step(action)
            value, log_probs, entropy = self.model.evaluate_action(state.unsqueeze(0).to(self.device), torch.tensor(action).to(self.device))

            loss = -log_probs * entropy
            loss.backward() # calc the gradients and store it in grad

            # Update Fisher information matrix
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if self.old_fisher is not None and name in self.old_fisher:
                        fisher[name] += self.ewc_gamma * self.old_fisher[name] + (1 - self.ewc_gamma) * (param.grad.data.clone() ** 2)
                    else:
                        fisher[name] += param.grad.data.clone() ** 2


            state = self.FloatTensor(next_state).to(self.device)
            
        for name in fisher:
            fisher[name] /= num_samples

        self.old_fisher = fisher.copy()
        return fisher

    def penalty(self, ewc_lambda:int, model: nn.Module):
        """Calculate the penalty to add to loss.

        Args:
            ewc_lambda (int): the lambda value
            model (nn.Module): The model which gets regulized (its the model, which traines and gets dynamically updated)

        Returns:
            _type_: float
        """
        loss = 0
        for n, p in model.named_parameters():
            loss += (self.fisher[n] * (p - self.mean_params[n]) ** 2).sum()
        return loss * ewc_lambda
    
    def update(self, agent, model, env_name):
        """Update the model, after learning the latest task. Here we calculate
        directly the FIM and also reset the mean_params.

        Args:
            agent: to get the new data (experience) of the latest run from the agents memory (current policy)
            model (_type_): _description_
            new_task (_type_): _description_
        """
        self.agent = agent
        self.env_name = env_name
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self.calculate_fisher()
        for n, p in deepcopy(self.params).items():
            self.mean_params[n] = variable(p.data)