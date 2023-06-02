from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, task: None, model: nn.Module, ewc_gamma=0.4, device=None):
        """The ewc algo
        Args:
            task_list (None): the task (in atari a env) for calculating the importance of task w.r.t the paramters
            model (nn.Module): the model which params are important to protect
            device (None): The cuda device
        """
        if torch.cuda.is_available():
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

        self.model = model
        self.task = task # in atari domain task == env
        self.device = device
        self.ewc_gamma = ewc_gamma

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.mean_params = {}
        self.old_fisher = None
        self.fisher = self.calculate_fisher() # calculate the importance of params for the previous task
        
        for n, p in deepcopy(self.params).items():
            self.mean_params[n] = variable(p.data)
            
    def calculate_fisher(self, num_samples=10000):
        self.model.eval()
        fisher = {}
        
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            fisher[n] = variable(p.data)
        
        print(f"Calculation of the task for the importance of each parameter: {self.task.spec.id}")
        state = self.FloatTensor(self.task.reset()).to(self.device)
        for _ in range(num_samples):
            # Calculate gradients
            self.model.zero_grad()
            
            action = self.model.act(state.unsqueeze(0).to(self.device))
            next_state, reward, done  = self.task.step(action)
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
    
    def update(self, model, new_task, num_samples=10000):
        """Update the model, after learning the latest task. Here we calculate
        directly the FIM and also reset the mean_params.

        Args:
            model (_type_): _description_
            new_task (_type_): _description_
            num_samples (int, optional): _description_. Defaults to 1.
        """
        self.task = new_task
        self.fisher = self.calculate_fisher(num_samples=num_samples)
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        for n, p in deepcopy(self.params).items():
            self.mean_params[n] = variable(p.data)