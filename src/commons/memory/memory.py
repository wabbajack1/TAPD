import torch

class Memory(object):
    def __init__(self):
        self.states, self.actions, self.true_values = [], [], []

        if torch.cuda.is_available():
            FloatTensor = torch.cuda.FloatTensor
            LongTensor = torch.cuda.LongTensor
        else:
            FloatTensor = torch.FloatTensor
            LongTensor = torch.LongTensor
    
    def push(self, state, action, true_value):
        self.states.append(state)
        self.actions.append(action)
        self.true_values.append(true_value)
    
    def pop_all(self):
        states = torch.stack(self.states)
        actions = LongTensor(self.actions)
        true_values = FloatTensor(self.true_values).unsqueeze(1)
        
        self.states, self.actions, self.true_values = [], [], []
        
        return states, actions, true_values