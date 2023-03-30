from ProgNetAbstract import ProgColumnGenerator
from blocks import ProgDenseBlock, ProgConv2DBlock, MultiProgDenseBlock
from ProgNetAbstract import ProgNet, ProgColumn, ProgBlock
import torch
from EWC import EWC
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from copy import deepcopy
import torch.nn as nn

# hyper params
hidden_size = 512


class Actor(ProgColumnGenerator):
    def __init__(self):
        self.ids = 0

    """def generateColumn(self, parentCols, msg = None):
        b1 = ProgDenseBlock(28*28, 400, 0)
        b2 = ProgDenseBlock(400, 400, len(parentCols))
        b3 = ProgDenseBlock(400, 10, len(parentCols), activation=None)
        column = ProgColumn(self.__genID(), [b1, b2, b3], parentCols = parentCols)
        return column"""
    def generateColumn(self, parentCols):
        params_b1 = {"stride": 4}
        params_b2 = {"stride": 2}
        params_b3 = {"stride": 1}

        b1 = ProgConv2DBlock(4, 32, kernelSize=8,numLaterals=0, layerArgs=params_b1)
        b2 = ProgConv2DBlock(32, 64, kernelSize=4, numLaterals=len(parentCols), layerArgs=params_b2)
        b3 = ProgConv2DBlock(64, 64, kernelSize=3, numLaterals=len(parentCols), layerArgs=params_b3)
        b4 = ProgDenseBlock(3136, 200, numLaterals=len(parentCols) , activation=None, after_conv=True)
        b5 = ProgDenseBlock(200, 10, numLaterals=len(parentCols) , activation=None)
        column = ProgColumn(self.__genID(), [b1, b2, b3, b4, b5], parentCols = parentCols)
        return column

    def __genID(self):
        ids = self.ids
        self.ids += 1
        return ids

class Critic(ProgColumnGenerator):
    def __init__(self):
        self.ids = 0

    """def generateColumn(self, parentCols, msg = None):
        b1 = ProgDenseBlock(28*28, 400, 0)
        b2 = ProgDenseBlock(400, 400, len(parentCols))
        b3 = ProgDenseBlock(400, 10, len(parentCols), activation=None)
        column = ProgColumn(self.__genID(), [b1, b2, b3], parentCols = parentCols)
        return column"""
    def generateColumn(self, parentCols):
        params_b1 = {"stride": 4}
        params_b2 = {"stride": 2}
        params_b3 = {"stride": 1}

        b1 = ProgConv2DBlock(4, 32, kernelSize=8,numLaterals=0, layerArgs=params_b1)
        b2 = ProgConv2DBlock(32, 64, kernelSize=4, numLaterals=len(parentCols), layerArgs=params_b2)
        b3 = ProgConv2DBlock(64, 64, kernelSize=3, numLaterals=len(parentCols), layerArgs=params_b3)
        b4 = ProgDenseBlock(3136, 200, numLaterals=len(parentCols) , activation=None, after_conv=True)
        b5 = ProgDenseBlock(200, 1, numLaterals=len(parentCols) , activation=None)
        column = ProgColumn(self.__genID(), [b1, b2, b3, b4, b5], parentCols = parentCols)
        return column

    def __genID(self):
        ids = self.ids
        self.ids += 1
        return ids

class ActorCritic(ProgColumnGenerator):
    def __init__(self):
        self.ids = 0

    """def generateColumn(self, parentCols, msg = None):
        b1 = ProgDenseBlock(28*28, 400, 0)
        b2 = ProgDenseBlock(400, 400, len(parentCols))
        b3 = ProgDenseBlock(400, 10, len(parentCols), activation=None)
        column = ProgColumn(self.__genID(), [b1, b2, b3], parentCols = parentCols)
        return column"""
    def generateColumn(self, parentCols):
        params_b1 = {"stride": 4}
        params_b2 = {"stride": 2}
        params_b3 = {"stride": 1}

        b1 = ProgConv2DBlock(1, 32, kernelSize=8,numLaterals=0, layerArgs=params_b1)
        b2 = ProgConv2DBlock(32, 64, kernelSize=4, numLaterals=len(parentCols), layerArgs=params_b2)
        b3 = ProgConv2DBlock(64, 64, kernelSize=3, numLaterals=len(parentCols), layerArgs=params_b3)
        b4 = ProgDenseBlock(3136, 200, numLaterals=len(parentCols), after_conv=True)
        b5 = MultiProgDenseBlock(200, 10, numLaterals=len(parentCols))
        column = ProgColumn(self.__genID(), [b1, b2, b3, b4, b5], parentCols = parentCols)
        return column

    def __genID(self):
        ids = self.ids
        self.ids += 1
        return ids

class ActorCritic2(ProgColumnGenerator):
    def __init__(self):
        self.ids = 0

    """def generateColumn(self, parentCols, msg = None):
        b1 = ProgDenseBlock(28*28, 400, 0)
        b2 = ProgDenseBlock(400, 400, len(parentCols))
        b3 = ProgDenseBlock(400, 10, len(parentCols), activation=None)
        column = ProgColumn(self.__genID(), [b1, b2, b3], parentCols = parentCols)
        return column"""
    def generateColumn(self, parentCols):
        params_b1 = {"stride": 4}
        params_b2 = {"stride": 2}
        params_b3 = {"stride": 1}

        b1 = ProgDenseBlock(500, 200, numLaterals=len(parentCols))
        b2 = ProgDenseBlock(200, 100, numLaterals=len(parentCols))
        b3 = MultiProgDenseBlock(100, 10, numLaterals=len(parentCols))
        column = ProgColumn(self.__genID(), [b1, b2, b3], parentCols = parentCols)
        return column

    def __genID(self):
        ids = self.ids
        self.ids += 1
        return ids



class CriticDense(ProgColumnGenerator):
    def __init__(self):
        self.ids = 0

    def generateColumn(self, parentCols, msg = None):
        b1 = ProgDenseBlock(4, 128, 0)
        b2 = ProgDenseBlock(128, 128, len(parentCols))
        b3 = ProgDenseBlock(128, 1, len(parentCols), activation=None)
        column = ProgColumn(self.__genID(), [b1, b2, b3], parentCols = parentCols)
        return column

    def __genID(self):
        ids = self.ids
        self.ids += 1
        return ids

class ActorDense(ProgColumnGenerator):
    def __init__(self):
        self.ids = 0

    def generateColumn(self, parentCols, msg = None):
        b1 = ProgDenseBlock(4, 128, 0)
        b2 = ProgDenseBlock(128, 128, len(parentCols))
        b3 = ProgDenseBlock(128, 2, len(parentCols), activation=None)
        column = ProgColumn(self.__genID(), [b1, b2, b3], parentCols = parentCols)
        return column

    def __genID(self):
        ids = self.ids
        self.ids += 1
        return ids

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)


        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=3136, out_features=10)

    def forward(self, x):
        # Pass the input through the convolutional layers
        x = self.conv1(x)
        x = nn.functional.relu(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)

        x = self.conv3(x)
        x = nn.functional.relu(x)

        # Flatten the output of the convolutional layers
        x = x.view(-1, 3136)

        # Pass the flattened output through the fully connected layers
        x = self.fc1(x)

        return x


if __name__ == "__main__":

    """# create nets
    actor = ProgNet(colGen = Actor())
    critic = ProgNet(colGen = Critic())

    # init actor nets
    net1_actor = actor.addColumn()
    net2_actor = actor.addColumn()

    # init critic nets
    net1_critic = critic.addColumn()
    net2_critic = critic.addColumn()

    # register parameters
    optimizer_actor = torch.optim.SGD(actor.parameters(), lr=0.8, momentum=0.9)
    optimizer_critic = torch.optim.SGD(critic.parameters(), lr=0.8, momentum=0.9)"""

    net = ProgNet(colGen = ActorCritic())
    net1 = net.addColumn()
    net2 = net.addColumn()

    optimizer_ac = torch.optim.SGD(net.parameters(), lr=0.8, momentum=0.9)
    loss = torch.nn.MSELoss()
    
    ################# training protocol #################
    #actor.train()
    #critic.train()
    net.train()

    print("============= before update =============")

    print("============= before  =============")
    for name, param in net.state_dict().items():
        #print(name, param.size(), param.requires_grad)
        if name == "columns.0.blocks.0.module.weight":
            prev_param_actor = deepcopy(param)
            print(name, param.size(), param.grad, param.grad_fn, "\n")
    
    """print("============= critic =============")
    for name, param in critic.state_dict().items():
        #print(name, param.size(), param.requires_grad)
        if name == "columns.0.blocks.0.module.weight":
            prev_param_critic = deepcopy(param)
            print(name, param.size(), param.grad, param.grad_fn, "\n")"""
    # training loop
    for i in range(1):
        """optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        actor.cuda()
        critic.cuda()"""

        x = torch.randn(7, 1, 84, 84)
        #x = x.cuda()
        target1 = torch.randn((7, 10))
        target2 = torch.randn((7, 1))
        #ac = actor(net1_actor, x)
        #val = critic(net1_critic, x)
        #ac, val = net(net2, x)
        z, u = net(net2, x)
        print(f"z {z.shape}, u {u.shape}")
        diff1_actor = loss(z, target1)
        diff2_actor = loss(u, target2)
        #diff2_critic = loss(val, target2)
        diff = diff1_actor + diff2_actor
        #actor.freezeColumn(net1_actor)
        #critic.freezeColumn(net1_critic)
        diff.backward()
        #make_dot(diff, params=dict(net.named_parameters())).view()
        """print("============= gradients  =============")

        print("============= actor =============")
        for name, param in actor.named_parameters():
            if param.requires_grad:
                print(f"-> gradient {param.grad is not None} -- {name} -- {param.size()}")

        print("============= critic =============")
        for name, param in critic.named_parameters():
            if param.requires_grad:
                print(f"-> gradient {param.grad is not None} -- {name} -- {param.size()}")"""

        for name, param in net.named_parameters():
            if param.requires_grad:
                print(f"-> gradient {param.grad is not None} -- {name} -- {param.size()}")

        print(diff1_actor)

        """optimizer_actor.step()
        optimizer_critic.step()"""
        optimizer_ac.step()
        optimizer_ac.zero_grad()
    
    """
    print("============= after update =============")

    print("============= actor =============")
    for name, param in actor.state_dict().items():
        #print(name, param.size())
        if name == "columns.0.blocks.0.module.weight":
            print(torch.equal(param, prev_param_actor))
            print(name, param.size(), param.grad, param.requires_grad)

    print("============= critic =============")
    for name, param in critic.state_dict().items():
        #print(name, param.size())
        if name == "columns.0.blocks.0.module.weight":
            print(torch.equal(param, prev_param_critic))
            print(name, param.size(), param.grad, param.requires_grad)
    """