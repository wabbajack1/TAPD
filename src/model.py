import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from src.distributions import Bernoulli, Categorical, DiagGaussian
from src.utils import init, custom_init
from copy import deepcopy

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BigPolicy(nn.Module):
    """The BigPolicy takes in two seperate policies, to combine them into one
    big Policy. This means that the two seperate pre specified policies (which can have
    the same architecture and logic) compute together. This means, the lateral/hidden
    output of the first given policy are the input of the second given policy, i.e. building
    a brigde between two policies. 

    Args:
        nn (_type_): _description_
    """
    def __init__(self, policy_a, policy_b, adaptor, use_lateral_connection=False):
        super(BigPolicy, self).__init__()
        self.policy_a = deepcopy(policy_a) # kb column
        self.policy_b = policy_b # active column
        self.adaptor = adaptor
        self.use_lateral_connection = use_lateral_connection
        self.experience = 0

        try:
            wandb.watch(self, log_freq=100, log="all", log_graph=True)
        except:
            pass

    def update_model(self, policy):
        self.policy_a.load_dict_state(policy.state_dict())

    def forward(self, inputs, action=None):
        # use policy_a to get the lateral connections
        a1, a2, _, a3 = self.policy_a(inputs)

        # use adaptor, after seeing one experience/task from inputs of the policy_b
        if self.use_lateral_connection:
            x1, x2, x3 = self.adaptor(a1, a2, a3)
            b1 = self.policy_b.base.main[:2](inputs)
            b1 = b1 + x1

            b2 = self.policy_b.base.main[2:4](b1)
            b2 = b2 + x2

            b3 = self.policy_b.base.main[4:](b2)
            b3 = b3 + x3
            critic_b3 = self.policy_b.base.critic_linear(b3)
        else:
            _, _, critic_b3, b3 = self.policy_b(inputs)


        return critic_b3, b3

    def act(self, inputs, deterministic=False):
        value, actor_features = self.forward(inputs)
        dist = self.policy_b.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        # print(f"Input shape {inputs.shape}, output shape {action_log_probs.shape, value.shape, action.shape}")
        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _ = self.forward(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.forward(inputs)
        dist = self.policy_b.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        self.obs_shape = obs_shape
        self.action_space = action_space
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(self.obs_shape) == 3:
                base = CNNBase
            elif len(self.obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(self.obs_shape[0], **base_kwargs)

        if self.action_space.__class__.__name__ == "Discrete":
            num_outputs = self.action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif self.action_space.__class__.__name__ == "Box":
            num_outputs = self.action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif self.action_space.__class__.__name__ == "MultiBinary":
            num_outputs = self.action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs):
        x1, x2, value, actor_features = self.base(inputs)
        return x1, x2, value, actor_features

    def act(self, inputs, deterministic=False):
        _, _, value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        # print(f"Input shape {inputs.shape}, output shape {action_log_probs.shape, value.shape, action.shape}")
        return value, action, action_log_probs

    def get_value(self, inputs):
        _, _, value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        _, _, value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy
    
    def reset_weights(self):
        print("\nRESETTING WEIGHTS")
        self.apply(custom_init)
        # for name, m in self.named_modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         m.reset_parameters()
        #         print(f"Name of Tensor {name}: {m}")

class NNBase(nn.Module):
    def __init__(self, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = False

        # if recurrent:
        #     self.gru = nn.GRU(recurrent_input_size, hidden_size)
        #     for name, param in self.gru.named_parameters():
        #         if 'bias' in name:
        #             nn.init.constant_(param, 0)
        #         elif 'weight' in name:
        #             nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class CNNBase(NNBase):
    def __init__(self, num_inputs, hidden_size=256):
        super(CNNBase, self).__init__(hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 16, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(16, 32, 4, stride=2)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 9 * 9, hidden_size)), nn.ReLU()
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs):
        x1 = self.main[:2](inputs)
        x2 = self.main[2:4](x1)
        x3 = self.main[4:](x2)

        return x1, x2, self.critic_linear(x3), x3

class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

class SingleLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SingleLayerMLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x
    
class Adaptor(nn.Module):


    """_summary_ new

    Args:
        nn (_type_): _description_
    """
    def __init__(self, hidden_size=256):
        super(Adaptor, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        
        self.conv1_adaptor = init_(nn.Conv2d(16, 16, kernel_size=1))
        self.conv2_adaptor = init_(nn.Conv2d(32, 32, kernel_size=1))
        
        self.single_layer_mlp = nn.Sequential(
            Flatten(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.train()

    def forward(self, x1, x2, x3):
        x1 = self.conv1_adaptor(x1)
        x2 = self.conv2_adaptor(x2)
        x3 = self.single_layer_mlp(x3)

        return x1, x2, x3
    
    def reset_weights(self):
        print("\nRESETTING WEIGHTS")
        self.apply(custom_init)
        # for name, m in self.named_modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         m.reset_parameters()
        #         print(f"Name of Tensor {name}: {m}")

class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super(IntrinsicCuriosityModule, self).__init__()
        # self.feature_size = 32 * 7 * 7
        self.feature_size = 288
        num_inputs = obs_shape[0]

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        
        # self.conv = nn.Sequential(
        #     nn.Conv2d(num_inputs, 32, 8, stride=4), nn.ELU(),
        #     nn.Conv2d(32, 64, 4, stride=2), nn.ELU(),
        #     nn.Conv2d(64, 32, 3, stride=1), nn.ELU(),
        # )

        self.conv = nn.Sequential(nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                  nn.ReLU()
        )

        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('leaky_relu'))
        
        self.forward_net = nn.Sequential(
            nn.Linear(self.feature_size + num_actions, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.feature_size)
        )

    def forward(self, state, next_state, action):
        state_ft = self.conv(state)
        next_state_ft = self.conv(next_state)
        state_ft = state_ft.view(-1, self.feature_size)
        next_state_ft = next_state_ft.view(-1, self.feature_size)
        # print(state_ft.shape,next_state_ft.shape)
        # print(torch.cat((state_ft, action), 1).shape)
        return self.forward_net(torch.cat((state_ft, action), 1)), next_state_ft