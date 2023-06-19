import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a single-layer network
class PolicyNet(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(PolicyNet, self).__init__()
        self.fc = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        x = self.fc(x)
        print(x, x.shape)
        print(F.softmax(x, dim=1),F.softmax(x, dim=1).shape)
        return F.softmax(x, dim=1)

# Instantiate the network
n_inputs = 4  # number of input features
n_outputs = 2  # number of actions
policy_net = PolicyNet(n_inputs, n_outputs)

# Assume some inputs
inputs = torch.randn((3, n_inputs))

# Forward pass through the network to get the action probabilities
action_probs = policy_net(inputs)

# Choose an action (here, the first action for simplicity)
action = 0

# Compute the log probability of the action
log_prob = torch.log(action_probs[0, action])

# Compute the gradient of the log probability
policy_net.zero_grad()
log_prob.backward(retain_graph=True)

# Extract the gradients
gradients = torch.cat([p.grad.view(-1) for p in policy_net.parameters()])

# Compute the diagonal of the Fisher Information Matrix
fisher_diag = gradients ** 2

print('Fisher Information Matrix Diagonal:', fisher_diag)
