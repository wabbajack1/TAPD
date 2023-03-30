import torch


class Model(torch.nn.Module):
    def __init__(self, action_space, env):
        super(Model, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        feature_size = self.features(
            torch.zeros(1, *env.observation_space.shape)).cuda().view(1, -1).size(1)
        
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(feature_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(feature_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, action_space),
            torch.nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        value = self.critic(x)
        actions = self.actor(x)
        return value, actions
    
    def get_critic(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.critic(x)
    
    def evaluate_action(self, state, action):
        value, actor_features = self.forward(state)
        dist = torch.distributions.Categorical(actor_features)
        
        log_probs = dist.log_prob(action).view(-1, 1)
        entropy = dist.entropy().mean()
        
        return value, log_probs, entropy
    
    def act(self, state):
        value, actor_features = self.forward(state)
        dist = torch.distributions.Categorical(actor_features)
        
        chosen_action = dist.sample()
        return chosen_action.item()

import torch
import torch.nn as nn

class KB_Module(nn.Module):
    def __init__(self, device, env):
        super(Module1, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        feature_size = self.layer3(self.layer2(self.layer1(torch.zeros(1, *env.observation_space.shape)))).to(device).view(1, -1).size(1)

        self.critic = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        layer1_output = self.layer1(x)
        layer2_output = self.layer2(layer1_output)
        layer3_output = self.layer3(layer2_output)

        features_flatten = layer3_output.view(layer3_output.size(0), -1)
        value = self.critic(features_flatten)
        actions = self.actor(features_flatten)

        return value, actions, layer1_output, layer2_output, layer3_output

    def get_critic(self, x):
        """
        Get the critic output for the given input tensor.

        :param x: input tensor
        :return: critic output tensor
        """
        features_flatten = self.forward_features(x).view(x.size(0), -1)
        return self.critic(features_flatten)

    def evaluate_action(self, state, action):
        """
        Evaluate the action via the critic.

        :param state: state tensor
        :param action: action tensor
        :return: value, log_probs, entropy tensors
        """
        value, actor_features, _, _, _ = self.forward(state)
        dist = torch.distributions.Categorical(actor_features)

        log_probs = dist.log_prob(action).view(-1, 1)
        entropy = dist.entropy().mean()

        return value, log_probs, entropy

    def freeze_parameters(self):
        """
        Freeze all parameters of the model, so they won't be updated during backpropagation.
        """
        for param in self.parameters():
            param.requires_grad = False



class Active_Module(nn.Module):
    def __init__(self, device, env, lateral_connections=True):
        """
        Initialize the Module2 class.

        :param device: PyTorch device to run the model on (CPU or GPU)
        :param env: environment object
        :param lateral_connections: flag to enable or disable lateral connections (default: True)
        """
        super(Module2, self).__init__()

        # Define feature layers with lateral connections
        self.lateral_connections = lateral_connections

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        feature_size = self.layer3(self.layer2(self.layer1(torch.zeros(1, *env.observation_space.shape)))).to(device).view(1, -1).size(1)

        self.critic = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
            nn.Softmax(dim=-1)
        )

        if not self.lateral_connections:
            self.lateral_layer1_output = torch.zeros_like(self.layer1[0].weight)
            self.lateral_layer2_output = torch.zeros_like(self.layer2[0].weight)
            self.lateral_layer3_output = torch.zeros_like(self.layer3[0].weight)
            self.lateral_critic_output = torch.zeros_like(self.critic[-1].weight)
            self.lateral_actor_output = torch.zeros_like(self.actor[-1].weight)

    def forward(self, x, lateral_layer1_output=None, lateral_layer2_output=None, lateral_layer3_output=None, lateral_critic_output=None, lateral_actor_output=None):
        """
        Forward pass with lateral connections.

        :param x: input tensor
        :param lateral_layer1_output: output from the first lateral connection of Module1 (optional)
        :param lateral_layer2_output: output from the second lateral connection of Module1 (optional)
        :param lateral_layer3_output: output from the third lateral connection of Module1 (optional)
        :param lateral_critic_output: output from the critic lateral connection of Module1 (optional)
        :param lateral_actor_output: output from the actor lateral connection of Module1 (optional)
        :return: value and action tensors
        """
        if not self.lateral_connections:
            lateral_layer1_output = self.lateral_layer1_output
            lateral_layer2_output = self.lateral_layer2_output
            lateral_layer3_output = self.lateral_layer3_output
            lateral_critic_output = self.lateral_critic_output
            lateral_actor_output = self.lateral_actor_output

        layer1_output = self.layer1(x) + lateral_layer1_output
        layer2_output = self.layer2(layer1_output) + lateral_layer2_output
        layer3_output = self.layer3(layer2_output) + lateral_layer3_output

        features_flatten = layer3_output.view(layer3_output.size(0), -1)
        value = self.critic(features_flatten) + lateral_critic_output
        actions = self.actor(features_flatten) + lateral_actor_output
    
        return value, actions

    def get_critic(self, x, lateral_critic_output=None):
        """
        Get the critic output for the given input tensor.

        :param x: input tensor
        :param lateral_critic_output: output from the critic lateral connection of Module1 (optional)
        :return: critic output tensor
        """
        if not self.lateral_connections:
            lateral_critic_output = self.lateral_critic_output

        features_flatten = self.forward_features(x).view(x.size(0), -1)
        return self.critic(features_flatten) + lateral_critic_output

    def evaluate_action(self, state, action, *lateral_outputs):
        """
        Evaluate the action via the critic.

        :param state: state tensor
        :param action: action tensor
        :param lateral_outputs: optional lateral outputs from Module1
        :return: value, log_probs, entropy tensors
        """
        value, actor_features = self.forward(state, *lateral_outputs)
        dist = torch.distributions.Categorical(actor_features)

        log_probs = dist.log_prob(action).view(-1, 1)
        entropy = dist.entropy().mean()

        return value, log_probs, entropy


class Adaptor(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super(Adaptor, self).__init__()
        self.conv1_adaptor = nn.Conv2d(1, 32, kernel_size=1)
        self.conv2_adaptor = nn.Conv2d(32, 64, kernel_size=1)
        self.conv3_adaptor = nn.Conv2d(64, 64, kernel_size=1)
        self.critic_adaptor = nn.Linear(64 * 7 * 7, 512)
        self.actor_adaptor = nn.Linear(64 * 7 * 7, 512)

    def forward(self, x1, x2, x3, critic_input, actor_input):
        y1 = self.conv1_adaptor(x1)
        y2 = self.conv2_adaptor(x2)
        y3 = self.conv3_adaptor(x3)
        y4 = self.critic_adaptor(critic_input)
        y5 = self.actor_adaptor(actor_input)
        return y1, y2, y3, y4, y5  
