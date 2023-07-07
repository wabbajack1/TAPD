import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, states, actions, true_values):
        self.states = states
        self.actions = actions
        self.true_values = true_values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.true_values[idx]