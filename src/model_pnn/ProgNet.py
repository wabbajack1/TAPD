
## The credits for this file go to Maxwell J Jacobson for implementing Doric
## you can learn more details about Doric in this repository :
# https://github.com/arcosin/Doric


import torch.nn as nn
from src.distributions import Bernoulli, Categorical, DiagGaussian
import wandb

"""
Class that acts as the base building-blocks of ProgNets.
Includes a module (usually a single layer),
a set of lateral modules, and an activation.
"""
class ProgBlock(nn.Module):
    """
    Runs the block on input x.
    Returns output tensor or list of output tensors.
    """
    def runBlock(self, x):
        raise NotImplementedError

    """
    Runs lateral i on input x.
    Returns output tensor or list of output tensors.
    """
    def runLateral(self, i, x):
        raise NotImplementedError

    """
    Runs activation of the block on x.
    Returns output tensor or list of output tensors.
    """
    def runActivation(self, x):
        raise NotImplementedError



"""
A column representing one sequential ANN with all of its lateral modules.
Outputs of the last forward run are stored for child column laterals.
Output of each layer is calculated as:
y = activation(block(x) + sum(laterals(x)))
"""
class ProgColumn(nn.Module):
    def __init__(self, colID, blockList, parentCols = []):
        super().__init__()
        self.colID = colID
        self.isFrozen = False
        self.parentCols = parentCols
        self.blocks = nn.ModuleList(blockList)
        self.numRows = len(blockList)
        self.lastOutputList = []

    def freeze(self, unfreeze = False):
        if not unfreeze:    # Freeze params.
            self.isFrozen = True
            for param in self.parameters():   param.requires_grad = False
        else:               # Unfreeze params.
            self.isFrozen = False
            for param in self.parameters():   param.requires_grad = True

    def forward(self, input):
        outputs = []
        x = input
        print(f"======= column {self.colID}, input shape {x.shape} =======")
        for row, block in enumerate(self.blocks):
            currOutput = block.runBlock(x)
            if row == 0 or len(self.parentCols) < 1:
                y = block.runActivation(currOutput)
            else:
                for c, col in enumerate(self.parentCols):
                    # print(f"column {c}, shape {col.lastOutputList[row - 1].shape}")
                    temp = block.runLateral(c, col.lastOutputList[row - 1])
                    currOutput += temp
                    # print(f"Output shape after lateral {c}: {temp.shape}, currentoutput {currOutput.shape}")
                y = block.runActivation(currOutput)
            outputs.append(y)
            x = y
        self.lastOutputList = outputs
        return outputs[-1], outputs[-2] # return the output of the last layer and the output of the second last layer; the first two are None because we are aligning with the BigPolicy class


"""
A progressive neural network as described in Progressive Neural Networks (Rusu et al.).
Columns can be added manually or with a ProgColumnGenerator.
https://arxiv.org/abs/1606.04671
"""
class ProgNet(nn.Module):
    def __init__(self, colGen = None, output_size=4):
        super().__init__()
        self.columns = nn.ModuleList()
        self.numRows = None
        self.numCols = 0
        self.colMap = dict()
        self.colGen = colGen
        self.colShape = None
        self.output_size = output_size

        try:
            wandb.watch(self, log_freq=100, log="all", log_graph=True)
        except:
            pass

    def addColumn(self, device,col = None, msg = None):
        if not col:
            parents = [colRef for colRef in self.columns]
            col = self.colGen.generateColumn(device = device, parent_cols = parents)
        self.columns.append(col)
        self.colMap[col.colID] = self.numCols
        self.numRows = col.numRows
        self.numCols += 1

        # initialize the output distribution, i.e. the actor network
        hidden_size = self.columns[0].blocks[-2].outSize
        self.dist = Categorical(hidden_size, self.output_size)
        self.dist.linear.to(device=device)

        return col.colID

    def freezeColumn(self, id):
        col = self.columns[self.colMap[id]]
        col.freeze()

    def freezeAllColumns(self):
        for col in self.columns:
            col.freeze()

    def unfreezeColumn(self, id):
        col = self.columns[self.colMap[id]]
        col.freeze(unfreeze = True)

    def unfreezeAllColumns(self):
        for col in self.columns:
            col.freeze(unfreeze = True)

    def gradientUsageInfoColumn(self, id):
        col = self.columns[self.colMap[id]]
        return col.isFrozen

    def getColumn(self, id):
        col = self.columns[self.colMap[id]]
        return col

    def forward(self, id, x):

        colToOutput = self.colMap[id]
        for i, col in enumerate(self.columns):
            y_last, y_second_last = col(x)
            if i == colToOutput:
                return y_last, y_second_last # return the output of the last layer and the output of the second last layer

    def act(self, inputs, deterministic=False, idx=None):
        value, actor_features = self(idx, inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        # print(action, dist.probs, dist.probs.shape)
        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, dist.logits

    def get_value(self, inputs, idx=None):
        value, _ = self(idx, inputs)
        return value

    def evaluate_actions(self, inputs, action, idx=None):
        value, actor_features = self(idx, inputs)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)

        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, dist.logits

    




"""
Class that generates new ProgColumns using the method generateColumn.
The parentCols list will contain references to each parent column,
such that columns can access lateral outputs.
Additional information may be passed through the msg argument in
generateColumn and ProgNet.addColumn.
"""
class ProgColumnGenerator:
    def generateColumn(self, parentCols, msg = None):
        raise NotImplementedError