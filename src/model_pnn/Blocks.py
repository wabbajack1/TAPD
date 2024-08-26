
# we implement the blocks that we want our columns in PNN to have
## Some of the blocks have been implemented in Doric
## you can learn more details about Doric in this repository :
# https://github.com/arcosin/Doric


from .ProgNet import *
import torch.nn as nn
from src.utils import init
import numpy as np

"""
A ProgBlock containing a single fully connected layer (nn.Linear).
Activation function can be customized but defaults to nn.ReLU.
"""
class ProgDenseBlock(ProgBlock):
    def __init__(self, inSize, outSize, numLaterals, drop_out = 0, activation = nn.ReLU()):
        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.module = init_(nn.Linear(inSize, outSize))
        self.dropOut = nn.Dropout(drop_out)
        self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        self.dropOut_laterals = nn.Dropout(drop_out)
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

        self.blockType = "Dense"

    def runBlock(self, x):
        return self.dropOut(self.module(x))

    def runLateral(self, i, x):
        lat = self.laterals[i]
        return self.dropOut_laterals(lat(x))

    def runActivation(self, x):
        return self.activation(x)


# a class for implementing an LSTM block
# each block will only have one LSTM layer. if you want to stack LSTMs on each other, you can use separate blocks.
class ProgLSTMBlock(ProgBlock):
    def __init__(self, inSize, outSize, numLaterals, lateralsType = 'linear',drop_out = 0):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.module = nn.LSTM(input_size=inSize,hidden_size = outSize,num_layers=1,batch_first=True)
        self.dropOut = nn.Dropout(drop_out)
        self.lateralsType = lateralsType
        # the research paper is kind of obscure when it comes to defining the structure of the lateral layer
        # between h^t,(l-1) and h^(t+1),l.
        # in the example given out in the paper, a linear layer is applied to the output of the last block(from the previous column).
        # on the other hand, most of the implementations of PNN in different repositories tend to apply
        # an nn similarly structured with the current block to the incoming output from the previous block
        # hence, we will implement two types and experiment to find the best outcome:
        # 1) the lateral layer between the two layers is a linear layer
        # 2) the lateral layer between the two layers is an LSTM itself.
        if lateralsType == 'linear':
            self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        else:
            self.laterals = nn.ModuleList([nn.LSTM(input_size= inSize, hidden_size= outSize,num_layers=1,batch_first=True) for _ in range(numLaterals)])
            self.dropOut_laterals = nn.Dropout(0.2)
            # we set a dropout =0.2 to avoid  overfitting.
        self.activation = (lambda x: x) # we will be outputting the result as it is to the next blocks.

        self.blockType = "LSTM"
    
    def runBlock(self, x):
        out, (h, c) = self.module(x)
        return self.dropOut(out)

    def runLateral(self, i, x):
        lat = self.laterals[i]
        if self.lateralsType == 'linear':
            return lat(x)
        else:
            out,(h,c) = lat(x)
            return self.dropOut_laterals(out)

    def runActivation(self, x):
        return self.activation(x)
        
        
        
I_FUNCTION = (lambda x : x)

"""
A ProgBlock containing a single Conv2D layer (nn.Conv2d) with Batch Normalization.
Activation function can be customized but defaults to nn.ReLU.
Stride, padding, dilation, groups, bias, and padding_mode can be set with layerArgs.
"""



class ProgConv2DBNBlock(ProgBlock):
    def __init__(self, inSize, outSize, kernelSize, numLaterals,flatten = False, activation = nn.ReLU(), layerArgs = dict(), bnArgs = dict(), skipConn = False, lambdaSkip = I_FUNCTION):
        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain("relu"))

        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.skipConn = skipConn
        self.skipVar = None
        self.skipFunction = lambdaSkip
        self.kernSize = kernelSize
        self.flatten = flatten
        self.module = init_(nn.Conv2d(inSize, outSize, kernelSize, **layerArgs))
        
        self.moduleBN = nn.BatchNorm2d(outSize, **bnArgs)
        self.laterals = nn.ModuleList([nn.Conv2d(inSize, outSize, kernelSize, **layerArgs) for _ in range(numLaterals)])
        self.lateralBNs = nn.ModuleList([nn.BatchNorm2d(outSize, **bnArgs) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation
        
        # id that this block is a Conv2DBN block
        self.blockType = "Conv2D"

    def runBlock(self, x):
        if self.skipConn:
            self.skipVar = x
        if self.flatten:
            return self.module(x).view(x.shape[0], -1)
        return self.module(x)

    def runLateral(self, i, x):
        if self.flatten:
            return self.laterals[i](x).view(x.shape[0], -1)
        else:
            return self.laterals[i](x)

    def runActivation(self, x):
        if self.skipConn and self.skipVar is not None:
            x = x + self.skipFunction(self.skipVar)
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "Conv2DBN"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["kernel_size"] = self.kernSize
        data["skip"] = self.skipConn
        return data