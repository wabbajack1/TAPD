import torch
import wandb
import torch.nn as nn

class Model(nn.Module):
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
    
class KB_Module(nn.Module):
    def __init__(self, device):
        super(KB_Module, self).__init__()
        self.device = device

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

        feature_size = self.layer3(self.layer2(self.layer1(torch.zeros(1, *(1, 84, 84))))).to(device).view(1, -1).size(1)

        self.critic = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 18),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = x3.view(x3.size(0), -1)

        critic_output = self.critic(x4)
        actor_output = self.actor(x4)

        return x1, x2, x3, x4, critic_output, actor_output

    def act(self, state):
        _, _, _, _, critic_output, actor_features = self.forward(state.to(self.device))
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
        return critic_output

    def evaluate_action(self, state, action):
        """
        Evaluate the action via the critic.

        :param state: state tensor
        :param action: action tensor
        :return: value, log_probs, entropy tensors
        """
        _, _, _, _, value, actor_features = self.forward(state.to(self.device))
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

        feature_size = self.layer3(self.layer2(self.layer1(torch.zeros(1, *(1, 84, 84))))).to(device).view(1, -1).size(1)

        self.critic = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 18),
            nn.Softmax(dim=-1)
        )
        
        self.adaptor = Adaptor(feature_size) # init adaptor layers like in prognet


    def reinit_parameters(self, seed=None):
        """
        Reinitialize the parameters of the model. If a seed is given, use it to generate a deterministic initialization.

        :param seed: Seed for the random number generator
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            # When running on the CuDNN backend, two further options must be set
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        for layer in [self.layer1, self.layer2, self.layer3, self.critic, self.actor]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            else:
                for sublayer in layer:
                    if hasattr(sublayer, 'reset_parameters'):
                        sublayer.reset_parameters()
            
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
        
        return critic_output

    def act(self, state):
        value, actor_features = self.forward(state.to(self.device))
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


class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, forward_hidden_dim):
        super(ICM, self).__init__()

        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, forward_hidden_dim),
            nn.ReLU(),
            nn.Linear(forward_hidden_dim, state_dim),
        )

    def forward(self, state, action):
        # Forward model
        action_one_hot = F.one_hot(action, num_classes=predicted_action.shape[-1])
        state_action = torch.cat([state, action_one_hot], dim=-1)
        predicted_next_state = self.forward_model(state_action)

        return predicted_next_state

class ProgressiveNet(nn.Module):
    def __init__(self, model_a, model_b):
        super(ProgressiveNet, self).__init__()
        self.model_a = model_a
        self.model_b = model_b
        #self.icm = ICM(state_dim, action_dim, forward_hidden_dim)
        self.icm = None
        
        wandb.watch(self, log_freq=1, log="all")

    def forward(self, x, action=None):
        x1, x2, x3, x4, critic_output_model_a, actor_output_model_a = self.model_a(x)
        critic_output_model_b, actor_output_model_b = self.model_b(x, [x1, x2, x3, x4])
        
        if action is not None:
            predicted_next_state = self.icm(x, action)
            return (
                critic_output_model_b,
                actor_output_model_b,
                critic_output_model_a,
                actor_output_model_a,
                predicted_next_state
            )

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
