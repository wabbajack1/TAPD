import torch
import random

class Memory(object):
    def __init__(self):
        self.states, self.actions, self.true_values = [], [], []
        self.FloatTensor = torch.FloatTensor
        self.LongTensor = torch.LongTensor

    def shuffle_blocks(self, data, subset_size=32):
        # Split data into subsets
        subsets = [data[i:i+subset_size] for i in range(0, len(data), subset_size)]
        
        # Shuffle the subsets
        random.shuffle(subsets)
        
        # Recombine into one list
        shuffled_data = [item for subset in subsets for item in subset]
        
        return shuffled_data
    
    def push(self, state, action, true_value):
        self.states.append(state)
        self.actions.append(action)
        self.true_values.append(true_value)
    
    def pop_all(self):
        states = torch.stack(self.states)
        actions = self.LongTensor(self.actions)
        true_values = self.FloatTensor(self.true_values).unsqueeze(1)
        
        return states, actions, true_values
    
    def delete_memory(self):
        self.states, self.actions, self.true_values = [], [], []