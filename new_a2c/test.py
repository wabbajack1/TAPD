import torch
from torch.utils.data import DataLoader, TensorDataset

# Create synthetic dataset
data_size = 100*128
input_dim = 5
output_dim = 1

# Generate random input and output tensors
X = torch.randn(data_size, input_dim)
y = torch.randn(data_size, output_dim)

# Wrap the tensors using TensorDataset
dataset = TensorDataset(X, y)

# Create DataLoader
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Show length of DataLoader
# This will indicate how many batches are in the DataLoader
len_dataloader = len(dataloader)
print(f"Length of DataLoader: {len_dataloader}", f"value {100/len_dataloader}")

