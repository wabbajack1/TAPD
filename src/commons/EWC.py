from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from typing import Optional
from commons.memory.CustomDataset import CustomDataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils import parameters_to_vector

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, agent:None, model: nn.Module, ewc_lambda=175, ewc_gamma=0.4, batch_size_fisher=32, device=None, env_name:Optional[str] = None):
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
            
        self.ewc_start_timestep = 1000
        self.model = model
        self.device = device
        self.ewc_gamma = ewc_gamma
        self.ewc_lambda = ewc_lambda
        self.env_name = env_name
        self.agent = agent # we need the memory module of this object (in atari domain task == env == data)
        self.batch_size_fisher = batch_size_fisher

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad and "critic" not in n}
        self.mean_params = {}
        self.old_fisher = None
        self.fisher = self.calculate_fisher() # calculate the importance of params for the previous task
        self.mean_model = deepcopy(self.model)
        
        for n, p in deepcopy(self.params).items():
            self.mean_params[n] = variable(p.data)
    
    def calculate_fisher(self):
        print(f"Calculation of the task for the importance of each parameter: {self.env_name}")
        self.model.eval()
        
        fisher = {}
        for n, p in deepcopy(self.params).items():
            fisher[n] = variable(p.detach().clone().zero_())
        
        states, actions, true_values = self.agent.memory.pop_all()
        self.agent.memory.delete_memory()
        dataset = CustomDataset(states, actions, true_values)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        i = 0
        for batch_states, batch_actions, batch_true_values in dataloader:
            #print("Parameters:", len(dataset), len(dataloader), batch_states.shape, len(dataloader)/self.agent.no_of_workers, self.batch_size_fisher)
            # print("batch size ewc", batch_states.shape, batch_actions.shape, batch_true_values.shape)
            
            # Calculate gradients
            self.model.zero_grad()
            
            batch_states = batch_states.to(self.device)
            batch_actions = batch_actions.to(self.device)
            batch_true_values = batch_true_values.to(self.device)
            values, log_probs, entropy = self.model.evaluate_action(batch_states, batch_actions)
            
            values = torch.squeeze(values)
            log_probs = torch.squeeze(log_probs)
            entropy = torch.squeeze(entropy)
            batch_true_values = torch.squeeze(batch_true_values)
            
            advantages = batch_true_values - values
            # critic_loss = advantages.pow(2).mean()
            
            actor_loss = -(log_probs * advantages.detach()).mean()
            # total_loss = ((0.5 * critic_loss) + actor_loss - (0.01 * entropy)).backward()
            actor_loss.backward() # calc the gradients and store it in grad
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            
            # Update Fisher information matrix
            # y_t = a * y_t + (1-a)*y_{t-1}
            for name, param in self.model.named_parameters():
                if param.grad is not None and "critic" not in name:
                    if self.old_fisher is not None and name in self.old_fisher:
                        fisher[name] += self.ewc_gamma * self.old_fisher[name] + param.grad.detach().clone().pow(2)
                    else:
                        fisher[name] += param.grad.detach().clone().pow(2)
        
        for name in fisher:
            fisher[name].data = (fisher[name].data - torch.mean(fisher[name].data)) / torch.std(fisher[name].data + 1e-08)
            # fisher[name].data /= self.agent.no_of_workers
            print(name, fisher[name].data)
        
        self.old_fisher = fisher.copy()
        return fisher
    
    def penalty(self, model: nn.Module):
        """Calculate the penalty to add to loss.

        Args:
            ewc_lambda (int): the lambda value
            model (nn.Module): The model which gets regulized (its the model, which traines and gets dynamically updated)

        Returns:
            _type_: float
        """
        loss = 0
        fisher_sum = 0
        mean_params_sum = 0
        for n, p in model.named_parameters():
            if "critic" not in n:
                # fisher = torch.sqrt(self.fisher[n] + 1e-08)
                fisher = self.fisher[n]
                loss += (fisher * (p - self.mean_params[n]).pow(2)).sum()
                fisher_sum += self.fisher[n].sum()
                mean_params_sum += self.mean_params[n].sum()
                # print(n, torch.sqrt(self.fisher[n] + 1e-05))
        
        # euclidean_distance = compute_distance(model, self.mean_model, "euclidean")
        # cosine_similarity = compute_distance(model, self.mean_model, "cosine")
        
        # print(f"Euclidean Distance: {euclidean_distance}")
        # print(f"Cosine Similarity: {cosine_similarity}")
        print("EWC Loss", (self.ewc_lambda * loss).item(), f"EWC lambda {self.ewc_lambda}", f"Fisher: {fisher_sum.sum()}", f"mean params: {mean_params_sum.sum()}")
        return self.ewc_lambda * loss
    
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
            
            
def compute_distance(model1, model2, mode="euclidean"):
    params1 = parameters_to_vector(model1.parameters())
    params2 = parameters_to_vector(model2.parameters())

    if mode == "euclidean":
        return torch.norm(params1 - params2).item()
    elif mode == "cosine":
        return torch.nn.functional.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0)).item()
    else:
        raise ValueError(f"Unknown mode: {mode}")