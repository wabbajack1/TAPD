# The code used is adopted from the following work: https://github.com/khashiii97/Progressive-Neural-Networks-for-IDS/tree/master and https://github.com/arcosin/Doric
# Test the model_pnn module, wth to rl algorithms

from src.envs import make_vec_envs
from .ColumnGenerator import *
from .ProgNet import ProgNet
import torch

envs = make_vec_envs("PongNoFrameskip-v4", 1, 1, 0.99, "", "cpu", False)


channels = envs.observation_space.shape[0] # input channels for cnn

# model definition
column_generator = Column_generator_CNN(num_of_conv_layers=2, kernel_size=8, num_of_classes=4, num_dens_Layer=1)
pnn = ProgNet(column_generator)

# dataset
dataset = torch.randn(6400, channels, 84, 84)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

task_number = 10
visit = 10

for vis in range(visit):
    print(f"===== Visit {vis} =====")
    for task in range(task_number):
        if vis == 0:
            idx = pnn.addColumn(device="cpu") # add a column to the network for each task
            print(f"===== New column generated: {idx} =====")

        for i, data in enumerate(dataloader):
            final_output, second_final_output = pnn.forward(idx, data)
            print(final_output.shape, second_final_output.shape)
            break
        
        for key in pnn.colMap.keys():
            print(f"Column {key}: gradients_frozen={pnn.gradientUsageInfoColumn(key)}")
        
        pnn.freezeAllColumns()
        # print(pnn.colMap)
        # print(f"Columns: {pnn.columns}")


