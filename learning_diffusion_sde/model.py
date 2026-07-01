"""
model.py
========
Network architecture shared between scripts 07 (train) and 08+ (rollout).
Import FlowNet from here — do NOT import 07_train_network.py.
"""

import torch
import torch.nn as nn
import config


class FlowNet(nn.Module):
    """
    G(x_normalised, z) → ΔX

    MLP mapping (normalised SDE position, latent noise) → step increment.
    """

    def __init__(self, hidden_dim=config.HIDDEN_DIM, n_layers=config.N_LAYERS):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_z):
        return self.net(x_z)
