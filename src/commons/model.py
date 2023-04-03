import torch
import wandb

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

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = x3.view(x3.size(0), -1)
        
        critic_x1 = x4
        actor_x1 = x4

        critic_output = self.critic(critic_x1)
        actor_output = self.actor(actor_x1)

        return x1, x2, x3, x4, critic_output, actor_output

    def get_critic(self, x):
        """
        Get the critic output for the given input tensor.

        :param x: input tensor
        :return: critic output tensor
        """
        with torch.no_grad():
            x1, x2, x3, x4, critic_output, actor_output = self.forward(x)
        return critic_output

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
    def __init__(self, device, env, lateral_connections=False):
        """
        Initialize the Module2 class.

        :param device: PyTorch device to run the model on (CPU or GPU)
        :param env: environment object
        :param lateral_connections: flag to enable or disable lateral connections (default: True)
        """
        super(Active_Module, self).__init__()

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

        if self.lateral_connections:
            self.adaptor = Adaptor(feature_size)

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

            print("with lat")
        else:
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = x3.view(x3.size(0), -1)
            critic_x1 = self.critic[0:2](x4)
            actor_x1 = self.actor[0:2](x4)
            
            print("without lat")

        critic_output = self.critic[-1:](critic_x1)
        actor_output = self.actor[-2:](actor_x1)

        return critic_output, actor_output
            

    def get_critic(self, x):
        with torch.no_grad():
            critic_output, _ = self.forward(x)
        
        return critic_output

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


class Adaptor(nn.Module):
    """_summary_ new

    Args:
        nn (_type_): _description_
    """
    def __init__(self, feature_size):
        super(Adaptor, self).__init__()
        self.conv1_adaptor = nn.Conv2d(32, 32, kernel_size=1)
        self.conv2_adaptor = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3_adaptor = nn.Conv2d(64, 64, kernel_size=1)
        self.critic_adaptor = nn.Linear(feature_size, 512)
        self.actor_adaptor = nn.Linear(feature_size, 512)

    def forward(self, x1, x2, x3, x4):
        y1 = self.conv1_adaptor(x1)
        y2 = self.conv2_adaptor(x2)
        y3 = self.conv3_adaptor(x3)
        y4 = self.critic_adaptor(x4)
        y5 = self.actor_adaptor(x4)
        return y1, y2, y3, y4, y5


class ProgressiveNet(nn.Module):
    def __init__(self, model_a, model_b):
        super(ProgressiveNet, self).__init__()
        self.model_a = model_a
        self.model_b = model_b

        #wandb.watch(self, log_freq=1, log="all")

    def forward(self, x):
        x1, x2, x3, x4, critic_output_model_a, actor_output_model_a = self.model_a(x)
        critic_output_model_b, actor_output_model_b = self.model_b(x, [x1, x2, x3, x4])
        return critic_output_model_b, actor_output_model_b, critic_output_model_a, actor_output_model_a
    
    def get_critic(self, x):
        with torch.no_grad():
            ritic_output_model_b, _, _, _ = self.forward(x)
            
        return ritic_output_model_b
    
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


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import runner
    from torchviz import make_dot
    import copy

    env = runner.environment_wrapper("PongNoFrameskip-v4")
    nn1 = KB_Module("cpu", env)
    #nn2 = Adaptor()
    nn3 = Active_Module("cpu", env, lateral_connections=True)

    nn_super = ProgressiveNet(nn1, nn3)
    #print(nn_super)

    optimizer_ac1 = torch.optim.SGD(nn1.parameters(), lr=0.8, momentum=0.9)
    optimizer_ac2 = torch.optim.SGD(nn3.parameters(), lr=0.8, momentum=0.9)
    #optimizer_super = torch.optim.SGD(nn_super.parameters(), lr=0.8, momentum=0.9)

    loss = torch.nn.MSELoss()
    old_params = nn_super.store_parameters(nn_super) 
   
    for i in range(0, 2):
        optimizer_ac1.zero_grad()
        optimizer_ac2.zero_grad()

        if i == 1:
            nn_super.freeze_model("model_b")
            old_params = nn_super.store_parameters(nn_super.model_b)

        input = torch.randn(10, 1, 84, 84)
        target1 = torch.randn(10, 6)
        target2 = torch.randn(10, 1)

        y1, y2, y3, y4 = nn_super(input)

        loss1 = loss(y2, target1)
        loss2 = loss(y1, target2)
        total_loss = loss1 + loss2
        total_loss.backward()
        optimizer_ac1.step()
        #optimizer_ac2.step()

        if i == 1:
            if nn_super.compare_parameters(old_params, nn_super.model_b):
                print("Parameter values are the same.")
            else:
                print("Parameter values have changed.")



    #print(y1.shape, y2.shape, y3.shape, y4.shape)
    #print(x.shape, y.shape, critic_output.shape, actor_output.shape, x4.shape)

    
    pass
