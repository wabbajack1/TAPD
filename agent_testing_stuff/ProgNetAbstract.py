import torch.nn as nn
from copy import deepcopy

############# Column Generator ##############
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

    def __genID(self):
        raise NotImplementedError

########################## Abstract Block ##########################
"""
Class that acts as the base building-blocks of ProgNets.
Includes a module (usually a single layer),
a set of lateral modules, and an activation.
"""
class ProgBlock(nn.Module):
    def __init__(self, device:str=None):
        super().__init__()
        self.device = "cpu" if device is None else device

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

##################### Abstract ProgNet #########################
"""
A progressive neural network as described in Progressive Neural Networks (Rusu et al.).
Columns can be added manually or with a ProgColumnGenerator.
https://arxiv.org/abs/1606.04671
"""
class ProgNet(nn.Module):
    def __init__(self, colGen = None):
        super().__init__()
        self.columns = nn.ModuleList()
        self.numRows = None
        self.numCols = 0
        self.colMap = dict()
        self.colGen = colGen # generate the column of the same specifaction

    def addColumn(self, col = None):
        """Generate new column (network) in the prognet.

        Args:
            col (nn.Module, optional): can be used to create columns if self.colGen is not
            initiated i.e. if there is no init of the ProgColumnGenerator. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            int: return the index of the created column
        """
        if not col:
            if self.colGen is None:
                raise ValueError("No column or generator supplied.")
            parents = [colRef for colRef in self.columns] # check if there are any parent columns
            col = self.colGen.generateColumn(parents) # generate column
        
        # col is an object of the class ProgColumn
        self.columns.append(col) # add colum in prognet (if col not specified take the input as the col)
        

        if col.colID in self.colMap:
            raise ValueError("Column ID must be unique.")
        
        self.colMap[col.colID] = self.numCols
        
        # check if number of rows of the columns is the same (can be tested incrementaly)
        if self.numRows is None:
            self.numRows = col.numRows
        else:
            if self.numRows != col.numRows:
                raise ValueError("Each column must have equal number of rows.")
        self.numCols += 1

        return col.colID

    def freezeColumn(self, id):
        if id not in self.colMap:
            raise ValueError("No column with ID %s found." % str(id))
        col = self.columns[self.colMap[id]]
        col.freeze()

    def freezeAllColumns(self):
        for col in self.columns:
            col.freeze()

    def unfreezeColumn(self, id):
        if id not in self.colMap:
            raise ValueError("No column with ID %s found." % str(id))
        col = self.columns[self.colMap[id]]
        col.freeze(unfreeze = True)

    def unfreezeAllColumns(self):
        for col in self.columns:
            col.freeze(unfreeze = True)

    def isColumnFrozen(self, id):
        if id not in self.colMap:
            raise ValueError("No column with ID %s found." % str(id))
        col = self.columns[self.colMap[id]]
        return col.isFrozen

    def getColumn(self, id):
        if id not in self.colMap:
            raise ValueError("No column with ID %s found." % str(id))
        col = self.columns[self.colMap[id]]
        return col

    def forward(self, id, x):
        """The forward function for prog net

        Args:
            id (int): id of column (network id)
            x (tensor): input tensor

        Raises:
            ValueError: ProgNet with one column
            ValueError: no column with the id

        Returns:
            tensor: output for target space
        """
        if self.numCols <= 0:
            raise ValueError("ProgNet cannot be run without at least one column.")
        if id not in self.colMap:
            raise ValueError(f"No column with ID {str(id)} found.")
        colToOutput = self.colMap[id] # set the index of the output column
        for i, col in enumerate(self.columns):
            print(f"\ncol {i} forward")
            y = col(x)
            if i == colToOutput: # check if the net is the output column
                return y

    def getData(self):
        data = dict()
        data["cols"] = [c.getData() for c in self.columns]
        return data

    def weight_reset(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()
    
    def network_reset(self, id):
        self.getColumn(id).apply(self.weight_reset)

############# Column i.e. a NN #############
"""
A column representing one sequential ANN with all of its lateral modules.
Outputs of the last forward run are stored for child column laterals.
Output of each layer is calculated as:
y = activation(block(x) + sum(laterals(x)))
"""
class ProgColumn(nn.Module):
    """
    A column representing one sequential ANN with all of its lateral modules.
    Outputs of the last forward run are stored for child column laterals.
    Output of each layer is calculated as:
    y = activation(block(x) + sum(laterals(x)))
    colID -- A unique identifier for the column.
    blockList -- A list of ProgBlocks that will be run sequentially.
    parentCols -- A list of pointers to columns that will be laterally connectected.
                If the list is empty, the column is unlateralized.
    """
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
        outputs = [] # gather the outputs
        x = input
        for row, block in enumerate(self.blocks):
            #print(f"block {row, len(self.parentCols)} run")
            currOutput = block.runBlock(x) # current output of a block
            if row == 0 or len(self.parentCols) < 1:
                if block.mutliOutputBlock == True:
                    y = block.runActivation(currOutput[0])
                    value = block.runActivation(currOutput[1]) # new
                else:
                    y = block.runActivation(currOutput)
            else:
                for c, col in enumerate(self.parentCols):
                    #print(col.lastOutputList[row - 1].size(), self.colID)
                    if isinstance(currOutput, tuple):
                        literal_values =  block.runLateral(c, col.lastOutputList[row-1])
                        ac = currOutput[0] + literal_values[0]
                        val = currOutput[1] + literal_values[1]
                    else:
                        currOutput += block.runLateral(c, col.lastOutputList[row-1])

                if block.mutliOutputBlock == True:
                    y = block.runActivation(ac)
                    value = block.runActivation(val)
                else:
                    y = block.runActivation(currOutput)

            if block.mutliOutputBlock == True:
                outputs.append((y, value))
            else:
                outputs.append(y)
            x = y
        
        self.lastOutputList = outputs

        return outputs[-1]