"""
Neural network architectures for FlexPINN
"""

import torch
import torch.nn as nn
from config import device

def weights_init(m):
    """Xavier uniform initialization for linear layers"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

class layer(nn.Module):
    """Basic neural network layer with activation"""
    def __init__(self, n_in, n_out, activation):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x

class LargeNetwork(nn.Module):
    """Large feature extraction network"""
    def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb, activation=nn.Tanh()):
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(layer(dim_in, n_node, activation))
        for _ in range(n_layer):
            self.net.append(layer(n_node, n_node, activation))
        self.net.append(layer(n_node, dim_out, activation=None))
        self.ub = torch.tensor(ub, dtype=torch.float).to(device)
        self.lb = torch.tensor(lb, dtype=torch.float).to(device)
        self.net.apply(weights_init)

    def forward(self, x):
        out = x
        for layer in self.net:
            out = layer(out)
        return out

class SmallNetwork(nn.Module):
    """Small output network for individual variables"""
    def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb, activation=nn.Tanh()):
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(layer(dim_in, n_node, activation))
        for _ in range(n_layer):
            self.net.append(layer(n_node, n_node, activation))
        self.net.append(layer(n_node, dim_out, activation=None))
        self.ub = torch.tensor(ub, dtype=torch.float).to(device)
        self.lb = torch.tensor(lb, dtype=torch.float).to(device)
        self.net.apply(weights_init)

    def forward(self, x):
        out = x
        for layer in self.net:
            out = layer(out)
        return out

class CombinedNetwork(nn.Module):
    """Combined network with one large and multiple small networks"""
    def __init__(self):
        super().__init__()
        from config import ub, lb
        self.large_network = LargeNetwork(
            dim_in=7, dim_out=20, n_layer=8, n_node=60, ub=ub, lb=lb
        ).to(device)
        self.small_networks = nn.ModuleList([
            SmallNetwork(
                dim_in=20, dim_out=1, n_layer=2, n_node=10, ub=ub, lb=lb
            ).to(device) for _ in range(9)
        ])

    def forward(self, x):
        large_network_output = self.large_network(x)
        small_network_outputs = []
        for network in self.small_networks:
            small_network_output = network(large_network_output)
            small_network_outputs.append(small_network_output)
        return small_network_outputs