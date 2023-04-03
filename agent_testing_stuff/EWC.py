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
    def __init__(self, model: nn.Module, dataset: list):
        #print(f"---- EWC ----")
        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.mean_params = {}
        self.precision_matrices = self.diag_fisher()

        for n, p in deepcopy(self.params).items():
            self.mean_params[n] = variable(p.data)
            
    def diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for x in self.dataset:
            self.model.zero_grad()
            #print("---> ewc x", x)
            x = variable(torch.tensor(x))
            x.requires_grad_() # set gradients on for input, otherwise will get error in the comp. graph

            #y = variable(torch.tensor(y.view(-1)))
            #x = x.view(x.size(0), -1)

            #print(x.shape, y.shape)
            output, val = self.model(x)
            output = output.view(1, -1)
            #print("---->", output)
            #print("output shape is", output.shape)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=0), label)
            loss.backward()
            
            for i, (n, p) in enumerate(self.model.named_parameters()):
                if p.grad is not None:
                    precision_matrices[n] += ((p.grad.data.clone()**2) / len(self.dataset))
                #print("============\n", p.grad.data.clone()**2)
                
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            loss += (self.precision_matrices[n] * (p - self.mean_params[n]) ** 2).sum()
        return loss

class EWC_online(EWC):
    def __init__(self, model: nn.Module, dataset: list):
        print(f"---- EWC online ----")
        super(EWC_online, self).__init__(model, dataset)

    def diag_fisher_online(self, dataset):
        self.model.eval()
        for x, y in dataset:
            self.model.zero_grad()
            x = variable(torch.tensor(x))
            #y = variable(torch.tensor(y.view(-1)))
            #x = x.view(x.size(0), -1)

            #print(x.shape, y.shape)
            output = self.model(x).view(1, -1)
            #print("output shape is", output.shape)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                self.precision_matrices[n].data += ((p.grad.data.clone()**2) / len(dataset))
                #print("============\n", p.grad.data.clone()**2)

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            loss += (self.precision_matrices[n] * (p - self.mean_params[n]) ** 2).sum()
        return loss