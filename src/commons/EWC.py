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
    def __init__(self, env, model: nn.Module):
        if torch.cuda.is_available():
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

        self.model = model
        self.env = env

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.mean_params = {}
        self.fisher = self.calculate_fisher()

        for n, p in deepcopy(self.params).items():
            self.mean_params[n] = variable(p.data)
            
    def calculate_fisher(self, num_samples=1000):
        self.model.eval()
        fisher = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            fisher[n] = variable(p.data)

        for _ in range(num_samples):
            state = self.FloatTensor(env.reset())
            done = False
            while not done:
                action = self.model.act(state.unsqueeze(0))
                next_state, reward, done, _ = env.step(action)
                value, log_probs, entropy = self.model.evaluate_action(state, action)

                # Calculate gradients
                self.model.kb_optimizer.zero_grad()
                (-log_probs * entropy).backward()

                # Update Fisher information matrix
                for name, param in self.model.named_parameters():
                    if p.grad is not None:
                        fisher[name] += param.grad.data.clone() ** 2

                state = self.FloatTensor(next_state)

        for name in fisher:
            fisher[name] /= num_samples

        return fisher

    def penalty(self, ewc_lambda, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            loss += (self.fisher[n] * (p - self.mean_params[n]) ** 2).sum()
        return loss * ewc_lambda