import glob
import os

import torch
import torch.nn as nn

from src.envs import VecNormalize
from torch.nn import Module
from typing import NamedTuple, List
from torch import Tensor


class LayerAndParameter(NamedTuple):
    layer_name: str
    layer: Module
    parameter_name: str
    parameter: Tensor

# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)
    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    print(module)

    original_device = module.weight.device
    module.weight.data = module.weight.data.to('cpu')
    module.bias.data = module.bias.data.to('cpu')
    
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)

    module.weight.data = module.weight.data.to(original_device)
    module.bias.data = module.bias.data.to(original_device)
    return module

def custom_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def freeze_everything(model: Module, set_eval_mode: bool = True):
    if set_eval_mode:
        model.eval()

    for layer_param in get_layers_and_params(model):
        layer_param.parameter.requires_grad = False


def unfreeze_everything(model: Module, set_train_mode: bool = True):
    if set_train_mode:
        model.train()

    for layer_param in get_layers_and_params(model):
        layer_param.parameter.requires_grad = True

def get_layers_and_params(model: Module, prefix='') -> List[LayerAndParameter]:
    result: List[LayerAndParameter] = []
    for param_name, param in model.named_parameters(recurse=False):
        result.append(LayerAndParameter(
            prefix[:-1], model, prefix + param_name, param))

    layer_name: str
    layer: Module
    for layer_name, layer in model.named_modules():
        if layer == model:
            continue

        layer_complete_name = prefix + layer_name + '.'

        result += get_layers_and_params(layer, prefix=layer_complete_name)

    return result

def generate_normalized_tasks(samples_nmb=100, x_range=(0, 8)):
    # Define smooth step functions using sigmoid
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def task1(x):
        return sigmoid(10 * (x - 1)) - sigmoid(10 * (x - 4))

    def task2(x):
        return sigmoid(10 * (x - 4)) - sigmoid(10 * (x - 8))

    # Generate data for the plot
    x = np.linspace(x_range[0], x_range[1], samples_nmb)
    y1 = task1(x)
    y2 = task2(x)
    # y3 = task3(x)

    # Normalize the tasks to make their sum equal to 1 at each point
    sum_tasks = y1 + y2 
    y1_normalized = y1 / sum_tasks
    y2_normalized = y2 / sum_tasks

    return y1_normalized, y2_normalized

