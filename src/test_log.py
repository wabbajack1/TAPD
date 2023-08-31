import torch
from tensorboardX import SummaryWriter
from datetime import datetime
import torch
from commons.model import Active_Module

net = Active_Module("cpu")
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = './logs/' + current_time
# Create a summary writer
writer = SummaryWriter(log_dir=log_dir)

# Example of logging a scalar
x = torch.randn((4, 1, 84, 84))
print(net)
writer.add_graph(net, x)
for i in range(100):
    writer.add_scalar('y=2x', i * 2, i)

for i in range(100):
    writer.add_scalar('y=2x', i**2, i)

for i in range(100):
    writer.add_scalar('y=2x', i**2/3, i)

# Close the writer
writer.close()
