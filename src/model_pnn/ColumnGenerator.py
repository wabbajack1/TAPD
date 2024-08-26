from .Blocks import *
from .ProgNet import ProgColumnGenerator
from src.utils import init

# we define a class that generates an LSTM based columns for us
class Column_generator_LSTM(ProgColumnGenerator):
    def __init__(self,input_size,hidden_size,num_of_classes,num_LSTM_layer,num_dens_Layer,dropout = 0.2):
        self.ids = 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_of_classes = num_of_classes
        self.num_LSTM_layer = num_LSTM_layer
        self.num_dens_Layer = num_dens_Layer
        self.dropout = dropout
        
    def __genID(self):
        id = self.ids
        self.ids += 1
        return id

    def create_column(self,device):
        columns = []
        columns.append(ProgLSTMBlock(inSize=self.input_size, outSize=self.hidden_size, lateralsType='LSTM',
                                     numLaterals=0, drop_out=self.dropout))
        if self.num_LSTM_layer == 2:
            columns.append(ProgLSTMBlock(inSize=self.hidden_size, outSize=self.hidden_size, lateralsType='LSTM',
                                         numLaterals=0, drop_out=self.dropout))

        activation = nn.Softmax(dim=2)
        if self.num_dens_Layer == 0:
            columns.append(ProgDenseBlock(inSize=self.hidden_size, outSize=self.num_of_classes
                                          , activation=activation, numLaterals=0,drop_out = self.dropout))
        elif self.num_dens_Layer == 1:  # adding an extra dense layer between LSTM and output
            columns.append(ProgDenseBlock(inSize=self.hidden_size, outSize=32
                                          , numLaterals=0,drop_out = self.dropout))
            columns.append(ProgDenseBlock(inSize=32, outSize=self.num_of_classes
                                          , activation=activation, numLaterals=0,drop_out = self.ropout))
        elif self.num_dens_Layer == 2:  # adding two extra dense layer between LSTM and output
            columns.append(ProgDenseBlock(inSize=self.hidden_size, outSize=64
                                          , numLaterals=0,drop_out = self.dropout))
            columns.append(ProgDenseBlock(inSize=64, outSize=32
                                          , activation=activation, numLaterals=0,drop_out = self.dropout))
            columns.append(ProgDenseBlock(inSize=32, outSize=self.num_of_classes
                                          , activation=activation, numLaterals=0,drop_out = self.dropout))

        return ProgColumn(self.__genID(), columns, parentCols=[]).to(device)

    def generateColumn(self,device,parent_cols): #generates column with its parents connections
        new_column = self.create_column(device = device)
        # setting connections to previous columns
        for i in range(1,len(new_column.blocks)):
            for j in range(len(parent_cols)):
                new_column.blocks[i].laterals.append(nn.Linear(new_column.blocks[i].inSize, new_column.blocks[i].outSize).to(device))
        new_column.freeze(unfreeze = True)
        return new_column
        
        
# we define a class that generates an LSTM based columns for us
class Column_generator_CNN(ProgColumnGenerator):
    def __init__(self,num_of_conv_layers,kernel_size,num_of_classes,num_dens_Layer, stride = 4):
        self.ids = 0
        self.num_of_conv_layers = num_of_conv_layers
        self.kernel_size = kernel_size
        self.num_of_classes = num_of_classes
        self.num_dens_Layer = num_dens_Layer
        self.stride = stride

        self.init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain("relu"))

    def __genID(self):
        id = self.ids
        self.ids += 1
        return id

    def create_column(self,device):
        columns = []
        dropout = 0

        columns.append(ProgConv2DBNBlock(inSize=4, outSize=16,kernelSize=self.kernel_size,flatten = self.num_of_conv_layers == 1,
                                     numLaterals=0, layerArgs = {'stride': self.stride}))
        
        dense_input_size = 32 * 9 * 9

        if self.num_of_conv_layers == 2:
            columns.append(ProgConv2DBNBlock(inSize=16, outSize=32, kernelSize=self.kernel_size//2, flatten=True,
                                     numLaterals=0, layerArgs = {'stride': self.stride//2}))

            dense_input_size = 32 * 9 * 9

        # activation = nn.Softmax(dim=1)
        activation = None
        if self.num_dens_Layer == 0:
            columns.append(ProgDenseBlock(inSize=dense_input_size, outSize=self.num_of_classes
                                          , activation=activation, numLaterals=0,drop_out = dropout))
        
        elif self.num_dens_Layer == 1:  # adding an extra dense layer between LSTM and output
            columns.append(ProgDenseBlock(inSize=dense_input_size, outSize=256
                                          , numLaterals=0,drop_out = dropout))
            columns.append(ProgDenseBlock(inSize=256, outSize=self.num_of_classes
                                          ,activation=activation, numLaterals=0, drop_out = dropout))

        elif self.num_dens_Layer == 2:  # adding an extra dense layer between LSTM and output
            columns.append(ProgDenseBlock(inSize=dense_input_size, outSize=256
                                          , numLaterals=0, drop_out=dropout))
            columns.append(ProgDenseBlock(inSize=256, outSize=128
                                          ,numLaterals=0, drop_out=dropout))
            columns.append(ProgDenseBlock(inSize=128, outSize=self.num_of_classes
                                          , activation=activation, numLaterals=0,drop_out = dropout))
        elif self.num_dens_Layer == 3:  # adding an extra dense layer between LSTM and output
            columns.append(ProgDenseBlock(inSize = dense_input_size, outSize=256
                                          , numLaterals=0, drop_out=dropout))
            columns.append(ProgDenseBlock(inSize=256, outSize=128
                                          ,numLaterals=0, drop_out=dropout))
            columns.append(ProgDenseBlock(inSize=128, outSize=64
                                          ,numLaterals=0,drop_out = dropout))
            columns.append(ProgDenseBlock(64, outSize=self.num_of_classes
                                          , activation=activation, numLaterals=0, drop_out=dropout))


        return ProgColumn(self.__genID(), columns, parentCols=[]).to(device)

    def generateColumn(self,device,parent_cols): #generates column with its parents connections

        init_linear = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        init_conv = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain("relu"))
        
        new_column = self.create_column(device = device)
        new_column.parentCols = parent_cols # setting connections to parent columns
        
        # setting connections to previous columns
        for i in range(1,len(new_column.blocks)):
            for j in range(len(parent_cols)):
                if new_column.blocks[i].blockType == "Dense":
                    new_column.blocks[i].laterals.append(init_linear(nn.Linear(new_column.blocks[i].inSize, new_column.blocks[i].outSize)).to(device))
                else:
                    new_column.blocks[i].laterals.append(init_conv(nn.Conv2d(new_column.blocks[i].inSize, new_column.blocks[i].outSize, self.kernel_size//2, stride = self.stride//2)).to(device))
                    
                    # if we have the last conv2d block make it flatten
                    if i == len(new_column.blocks) - 1:
                        new_column.blocks[i].flatten = True

        new_column.freeze(unfreeze = True)
        return new_column
  





