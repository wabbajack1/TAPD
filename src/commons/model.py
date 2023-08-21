import torch
import wandb
import torch.nn as nn
import os
import numpy as np
import random

class BaseConv(nn.Module):
    def __init__(self, num_inputs):
        super(BaseConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(num_inputs, 64, 3, stride=2, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                  nn.ReLU()
                                  )

    def forward(self, x):
        return self.conv(x)
    
class KB_Module(nn.Module):
    def __init__(self, device):
        super(KB_Module, self).__init__()
        self.device = device
        feature_size = 1568

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # feature_size = self.layer3(self.layer2(self.layer1(torch.zeros(1, *(1, 84, 84))))).to(device).view(1, -1).size(1)

        self.critic = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 18),
            nn.Softmax(dim=-1)
        )

        self._initialize_weights()

    def set_seed(self, seed: int = 44) -> None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}\n")

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = x3.view(x3.size(0), -1)

        critic_output = self.critic(x4)
        actor_output = self.actor(x4)

        return x1, x2, x3, x4, critic_output, actor_output

    def act(self, state):
        _, _, _, _, critic_output, actor_features = self.forward(state)
        dist = torch.distributions.Categorical(actor_features)
        
        chosen_action = dist.sample()
        return chosen_action.item()

    def get_critic(self, x):
        """
        Get the critic output for the given input tensor.

        :param x: input tensor
        :return: critic output tensor
        """
        with torch.no_grad():
            x1, x2, x3, x4, critic_output, actor_output = self.forward(x)
        return critic_output.to("cpu")

    def evaluate_action(self, state, action):
        """
        Evaluate the action via the critic.

        :param state: state tensor
        :param action: action tensor
        :return: value, log_probs, entropy tensors
        """
        _, _, _, _, value, actor_features = self.forward(state)
        dist = torch.distributions.Categorical(actor_features)

        log_probs = dist.log_prob(action).view(-1, 1)
        entropy = dist.entropy().mean()
        # print("Shape", log_probs.shape, entropy.shape, value.shape)

        return value, log_probs, entropy

    def freeze_parameters(self):
        """
        Freeze all parameters of the model, so they won't be updated during backpropagation.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        """
        Freeze all parameters of the model, so they won't be updated during backpropagation.
        """
        for param in self.parameters():
            param.requires_grad = True

class Active_Module(nn.Module):
    def __init__(self, device, lateral_connections:bool()=False):
        """
        Initialize the Module2 class.

        :param device: PyTorch device to run the model on (CPU or GPU)
        :param env: environment object
        :param lateral_connections: flag to enable or disable lateral connections (default: True)
        """
        super(Active_Module, self).__init__()
        # Define feature layers with lateral connections
        self.lateral_connections = lateral_connections
        self.device = device
        feature_size = 1568

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # feature_size = self.layer3(self.layer2(self.layer1(torch.zeros(1, *(1, 84, 84))))).to(device).view(1, -1).size(1)

        self.critic = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 18),
            nn.Softmax(dim=-1)
        )
        
        self.adaptor = Adaptor(feature_size) # init adaptor layers like in prognet

        self._initialize_weights()

    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def reset_weights(self, seed=None):
        print("===== RESET WEIGHTS =====")
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            # When running on the CuDNN backend, two further options must be set
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Set a fixed value for the hash seed
            os.environ["PYTHONHASHSEED"] = str(seed)
            print(f"Random seed set as {seed}\n")

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
                #print(f"Name of Tensor {name}: {m})

    def forward(self, x, previous_out_layers=None):

        """
        self.conv1_adaptor = nn.Conv2d(32, 64, kernel_size=1)
        self.conv2_adaptor = nn.Conv2d(64, 64, kernel_size=1)
        self.critic_adaptor = nn.Linear(64 * 7 * 7, 512)
        self.actor_adaptor = nn.Linear(64 * 7 * 7, 512)
        """

        x1 = self.layer1(x)

        if self.lateral_connections:
            y1, y2, y3, y4, y5 = self.adaptor(*previous_out_layers)
            x1 = x1 + y1
            
            x2 = self.layer2(x1)
            x2 = x2 + y2
            
            x3 = self.layer3(x2)
            x3 = x3 + y3

            x4 = x3.view(x3.size(0), -1)

            critic_x1 = self.critic[0:2](x4) + y4
            actor_x1 = self.actor[0:2](x4) + y5
        else:
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = x3.view(x3.size(0), -1)
            critic_x1 = self.critic[0:2](x4)
            actor_x1 = self.actor[0:2](x4)

        critic_output = self.critic[-1:](critic_x1)
        actor_output = self.actor[-2:](actor_x1)

        return critic_output, actor_output
            
    def get_critic(self, x):
        with torch.no_grad():
            critic_output, _ = self.forward(x)
        
        return critic_output.to("cpu")

    def act(self, state):
        value, actor_features = self.forward(state)
        dist = torch.distributions.Categorical(actor_features)
        
        chosen_action = dist.sample()
        return chosen_action.item()

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

    def freeze_parameters(self):
        """
        Freeze all parameters of the model, so they won't be updated during backpropagation.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        """
        Freeze all parameters of the model, so they won't be updated during backpropagation.
        """
        for param in self.parameters():
            param.requires_grad = True

class Adaptor(nn.Module):
    """_summary_ new

    Args:
        nn (_type_): _description_
    """
    def __init__(self, feature_size):
        super(Adaptor, self).__init__()
        self.conv1_adaptor = nn.Conv2d(32, 32, kernel_size=1)
        self.conv2_adaptor = nn.Conv2d(32, 32, kernel_size=1)
        self.conv3_adaptor = nn.Conv2d(32, 32, kernel_size=1)
        
        self.critic_adaptor = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU())
        
        self.actor_adaptor = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU())
        
        self._initialize_weights()

    def forward(self, x1, x2, x3, x4):
        y1 = self.conv1_adaptor(x1)
        y2 = self.conv2_adaptor(x2)
        y3 = self.conv3_adaptor(x3)
        y4 = self.critic_adaptor(x4)
        y5 = self.actor_adaptor(x4)
        return y1, y2, y3, y4, y5
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

class ICM(nn.Module):
    def __init__(self, num_inputs, num_actions=18):
        super(ICM, self).__init__()
        self.conv = BaseConv(num_inputs)
        self.state_dim = 64 * 6 * 6
        self.num_actions = num_actions

        self.inverse_net = nn.Sequential(
            nn.Linear(self.state_dim * 2, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, num_actions)
        )

        self.forward_net = nn.Sequential(
            nn.Linear(self.state_dim + self.num_actions, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.state_dim),
        )

    def one_hot(self, action):
        action_oh = torch.zeros((1, self.num_actions))  # one-hot action
        action_oh[0, action] = 1
        return action_oh

    def forward(self, state, next_state, action):
        # Forward model
        phi_state = self.conv(state)
        phi_next_state = self.conv(next_state)
        phi_state = phi_state.view(-1, self.state_dim)
        phi_next_state = phi_next_state.view(-1, self.state_dim)
        action_oh = self.one_hot(action)
        state_action = torch.cat([phi_state, action_oh], dim=1)
        state_nextstate = torch.cat([phi_state, phi_next_state], 1)

        return self.inverse_net(state_nextstate), self.forward_net(state_action), phi_next_state

class ProgressiveNet(nn.Module):
    def __init__(self, model_a, model_b):
        super(ProgressiveNet, self).__init__()
        self.model_a = model_a
        self.model_b = model_b
        # self.icm = ICM(1)
        self.icm = None
        
        try:
            wandb.watch(self, log_freq=1, log="all", log_graph=True)
        except:
            pass

    def forward(self, x, action=None):
        x1, x2, x3, x4, critic_output_model_a, actor_output_model_a = self.model_a(x)
        critic_output_model_b, actor_output_model_b = self.model_b(x, [x1, x2, x3, x4])
        
        if action is not None:
            pred_phi, phi = self.icm(x, action)
            return (
                critic_output_model_b,
                actor_output_model_b,
                critic_output_model_a,
                actor_output_model_a,
                pred_phi,
                phi
            )

        return critic_output_model_b, actor_output_model_b, critic_output_model_a, actor_output_model_a
    
    def get_critic(self, x):
        with torch.no_grad():
            ritic_output_model_b, _, _, _ = self.forward(x)
            
        return ritic_output_model_b.to("cpu")
    
    def check_calculated_gradient(self):
        skip1 = True
        skip2 = True
        for name, i in self.named_parameters():
            if "model_a" in name and skip1:
                skip1 = False
                print(f"======= {self.model_a._get_name()} ======")

            if "model_b" in name and skip2:
                skip2 = False
                print(f"======= {self.model_b._get_name()} ======")

            print(f"Gradients exists -> {i.grad is not None} for {name}")

    def evaluate_action(self, state, action):
        """
        Evaluate the action via the critic.

        :param state: state tensor
        :param action: action tensor
        :param lateral_outputs: optional lateral outputs from Module1
        :return: value, log_probs, entropy tensors
        """
        value, actor_features, _, _ = self.forward(state)
        dist = torch.distributions.Categorical(actor_features)

        log_probs = dist.log_prob(action).view(-1, 1)
        #print("----------log_probs", log_probs.shape)
        entropy = dist.entropy().mean()

        return value, log_probs, entropy

    def act(self, state):
        value, actor_features, _, _ = self.forward(state)
        dist = torch.distributions.Categorical(actor_features)
        chosen_action = dist.sample()
        return chosen_action.item()

    def freeze_model(self, model_name):
        if model_name == "model_a":
            model_to_freeze = self.model_a
        elif model_name == "model_b":
            model_to_freeze = self.model_b
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        for param in model_to_freeze.parameters():
            param.requires_grad = False

    @staticmethod
    def store_parameters(model):
        param_copy = {}
        for name, param in model.named_parameters():
            param_copy[name] = param.clone().detach()
        return param_copy

    @staticmethod
    def compare_parameters(old_params, model):
        same_values = True
        for name, param in model.named_parameters():
            if not torch.all(torch.eq(old_params[name], param)):
                same_values = False
                print(f"Parameter '{name}' has changed.")
        return same_values

