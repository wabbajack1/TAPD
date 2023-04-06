import torch

class Memory(object):
    def __init__(self, device):
        self.states, self.actions, self.true_values = [], [], []
        
        self.device = device
        self.FloatTensor = torch.FloatTensor
        self.LongTensor = torch.LongTensor
    
    def push(self, state, action, true_value):
        self.states.append(state)
        self.actions.append(action)
        self.true_values.append(true_value)
    
    def pop_all(self):
        states = torch.stack(self.states).to(self.device)
        actions = self.LongTensor(self.actions).to(self.device)
        true_values = self.FloatTensor(self.true_values).unsqueeze(1).to(self.device)
        
        self.states, self.actions, self.true_values = [], [], []
        
        return states, actions, true_values