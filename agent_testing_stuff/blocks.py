import torch.nn as nn
from ProgNetAbstract import ProgBlock
import torch

"""
A ProgBlock containing a single fully connected layer (nn.Linear).
Activation function can be customized but defaults to nn.ReLU.
"""
class ProgDenseBlock(ProgBlock):
    def __init__(self, inSize, outSize, numLaterals, activation = nn.ReLU(), after_conv=False):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.module = nn.Linear(inSize, outSize)
        self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        self.mutliOutputBlock = False
        self.after_conv = after_conv

        if activation is None:
            self.activation = (lambda x: x)
        else:                    
            self.activation = activation

        # flattening object for the input data if for example after a conv layer
        self.flatten = nn.Sequential(
            nn.Flatten()
        )

    def runBlock(self, x):
        if self.after_conv:
            x = self.flatten(x)
            #in_shape = x.shape[1]
            #self.module = nn.Linear(in_shape, self.outSize)

        # send to cuda
        """if self.device != "cpu":
            self.module.to("cuda:0")"""
        return self.module(x)

    def runLateral(self, i, x):
        """Run the literal connection of the block. Here every lateral gets optimized via brackprop algo.

        Args:
            i (int): the index of the literal (here all predecessor blocks connections at the specified layer)
            x (tensor): the output of the predecessor literal

        Returns:
            tensor: the output of the literal connection
        """

        # behavior of lateral connection 
        if self.after_conv:
            x = self.flatten(x)
            #in_shape = x.shape[1]
            #self.laterals = nn.ModuleList([nn.Linear(in_shape, self.outSize) for _ in range(self.numLaterals)])

        # send to cuda
        """if self.device != "cpu":
            for module in self.laterals:
                module.to('cuda:0')"""
        
        # choose the index of the lateral, i-th lateral is the connection from the i-th column
        lat = self.laterals[i]
        print("input is shape", x.shape)
        return lat(x)

    def runActivation(self, x):
        return self.activation(x)

"""
A ProgBlock containing a single fully connected layer (nn.Linear).
Activation function can be customized but defaults to nn.ReLU.

Specific for the last block of the A2C architecture.
"""
class MultiProgDenseBlock(ProgBlock):
    def __init__(self, inSize, outSize, numLaterals, activation = nn.ReLU(), after_conv=False):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.module1 = nn.Linear(inSize, outSize)
        self.module2 = nn.Linear(inSize, 1)
        self.laterals1 = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        self.laterals2 = nn.ModuleList([nn.Linear(inSize, 1) for _ in range(numLaterals)])
        self.mutliOutputBlock = True
        self.after_conv = after_conv

        self.flatten = nn.Sequential(
            nn.Flatten()
        )

        if activation is None:   
            self.activation = (lambda x: x)
        else:                    
            self.activation = activation

    def runBlock(self, x):
        if self.after_conv:
            x = self.flatten(x)
            """in_shape = x.shape[1]
            self.module1 = nn.Linear(in_shape, self.outSize)
            self.module2 = nn.Linear(in_shape, 1)"""
            
        """if self.device != "cpu":
            self.module1 = nn.Linear(in_shape, self.outSize)
            self.module2 = nn.Linear(in_shape, 1)
            self.module1.to('cuda:0')
            self.module2.to('cuda:0')"""

        action_logit = self.module1(x)
        value = self.module2(x)
        return action_logit, value

    def runLateral(self, i, x):
        if self.after_conv:
            x = self.flatten(x)
            """in_shape = x.shape[1]
            self.laterals1 = nn.ModuleList([nn.Linear(in_shape, self.outSize) for _ in range(self.numLaterals)])
            self.laterals2 = nn.ModuleList([nn.Linear(in_shape, 1) for _ in range(self.numLaterals)])"""

        """if self.device != "cpu":
            for module in self.laterals1:
                module.to('cuda:0')
            for module in self.laterals2:
                module.to('cuda:0')"""
                
        lat1 = self.laterals1[i]
        lat2 = self.laterals2[i]
        print("---->", lat1(x).shape, lat2(x).shape, x.shape)
        return lat1(x), lat2(x)

    def runActivation(self, x):
        return self.activation(x)


"""
A ProgBlock containing a single Conv2D layer (nn.Conv2d).
Activation function can be customized but defaults to nn.ReLU.
Stride, padding, dilation, groups, bias, and padding_mode can be set with layerArgs.
"""
class ProgConv2DBlock(ProgBlock):
    def __init__(self, inSize, outSize, kernelSize, numLaterals, activation = nn.ReLU(), layerArgs = dict()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.kernSize = kernelSize
        self.module = nn.Conv2d(inSize, outSize, kernelSize, **layerArgs)
        self.laterals = nn.ModuleList([nn.Conv2d(inSize, outSize, kernelSize, **layerArgs) for _ in range(numLaterals)])
        self.mutliOutputBlock = False
        if activation is None:   
            self.activation = (lambda x: x)
        else:                    
            self.activation = activation

    def runBlock(self, x):
        return self.module(x)

    def runLateral(self, i, x):
        lat = self.laterals[i]
        return lat(x)

    def runActivation(self, x):
        return self.activation(x)