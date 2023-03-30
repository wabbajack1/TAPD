
import torch
import torch.nn as nn
import wandb

class Model(nn.Module):
    def __init__(self, device, env):
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
            torch.zeros(1, *env.observation_space.shape)).to(device).view(1, -1).size(1)
        
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(feature_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(feature_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, env.action_space.n),
            torch.nn.Softmax(dim=-1)
        )

        wandb.watch(self, log_freq=1, log="all") # monitor paramters
    
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
        
class KB_Module(nn.Module):
    def __init__(self, device, env):
        super(KB_Module, self).__init__()

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

        wandb.watch(self, log_freq=1, log="all") # monitor paramters

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = x3.view(x3.size(0), -1)
        x4_critic = x4
        x4_actor = x4

        critic_output = self.critic(x4_critic)
        actor_output = self.actor(x4_actor)

        return critic_output, actor_output, x1, x2, x3

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
    
    def act(self, state):
        value, actor_features = self.forward(state)
        dist = torch.distributions.Categorical(actor_features)
        
        chosen_action = dist.sample()
        return chosen_action.item()
    
    def freeze_parameters(self):
        """
        Freeze all parameters of the model, so they won't be updated during backpropagation.
        """
        for param in self.parameters():
            param.requires_grad = False


class Active_Module(nn.Module):
    def __init__(self, device, env, lateral_connections=False):
        super(Active_Module, self).__init__()
        self.device = device
        self.lateral_connections = lateral_connections

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
        ).to(device)

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
        ).to(device)

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        ).to(device)

        feature_size = self.layer3(self.layer2(self.layer1(torch.zeros(1, *env.observation_space.shape).to(device)))).to(device).view(1, -1).size(1)

        self.critic = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ).to(device)

        self.actor = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
            nn.Softmax(dim=-1)
        ).to(device)

        if self.lateral_connections:
            self.adaptor = Adaptor()

        wandb.watch(self, log_freq=1, log="all") # monitor paramters

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = x3.view(x3.size(0), -1)

        if self.lateral_connections:
            y1, y2, y3, y4, y5 = self.adaptor(x1, x2, x3, x4, x4)
            x1 = x1 + y1
            x2 = x2 + y2
            x3 = x3 + y3
            x4_critic = x4 + y4
            x4_actor = x4 + y5
        else:
            x4_critic = x4
            x4_actor = x4

        critic_output = self.critic(x4_critic)
        actor_output = self.actor(x4_actor)

        return critic_output, actor_output

    def get_critic(self, x):
        """
        Get the critic output for the given input tensor. It also considers the 
        lateral_connections dependency.

        :param x: input tensor
        :return: critic output tensor
        """
        critic_output, _ = self.forward(x)
        return critic_output

    def forward_features(self, x):
        if not self.lateral_connections:
            lateral_layer1_output = self.lateral_layer1_output
            lateral_layer2_output = self.lateral_layer2_output
            lateral_layer3_output = self.lateral_layer3_output
        else:
            lateral_layer1_output, lateral_layer2_output, lateral_layer3_output = self.process_lateral_connections(x)

        layer1_output = self.layer1(x) + lateral_layer1_output
        layer2_output = self.layer2(layer1_output) + lateral_layer2_output
        layer3_output = self.layer3(layer2_output) + lateral_layer3_output

        flattened_output = layer3_output.view(layer3_output.size(0), -1)

        return flattened_output

    def process_lateral_connections(self, x):
        lateral_layer1_output = self.layer1(x)
        lateral_layer2_output = self.layer2(lateral_layer1_output)
        lateral_layer3_output = self.layer3(lateral_layer2_output)

        return lateral_layer1_output, lateral_layer2_output, lateral_layer3_output



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

    def act(self, state):
        value, actor_features = self.forward(state)
        dist = torch.distributions.Categorical(actor_features)
        
        chosen_action = dist.sample()
        return chosen_action.item()

    def freeze_parameters(self):
        """
        Freeze all parameters of the model, so they won't be updated during backpropagation.
        """
        for param in self.parameters():
            param.requires_grad = False


class Adaptor(nn.Module):
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