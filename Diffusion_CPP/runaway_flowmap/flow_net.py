"""
flow_net.py
===========

Flow Map neural network for Problem 1.

Learns the large-time transition

    (x, z) ---> Δx

where

    x : current particle position
    z : latent Gaussian noise

Output

    predicted increment Δx

The next position is

    x_next = x + Δx
"""

import torch
import torch.nn as nn


class FlowNet(nn.Module):
    """
    Fully-connected flow-map network.

    Input:
        x : (N,1)
        z : (N,1)

    Output:
        delta_x : (N,1)
    """

    def __init__(
        self,
        input_dim=2,
        hidden_dim=64,
        output_dim=1,
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, output_dim)

        )

    def forward(self, x, z):
        """
        Parameters
        ----------
        x : Tensor (N,1)

        z : Tensor (N,1)

        Returns
        -------
        delta_x : Tensor (N,1)
        """

        inp = torch.cat([x, z], dim=1)

        delta_x = self.network(inp)

        return delta_x


if __name__ == "__main__":

    model = FlowNet()

    print(model)

    x = torch.tensor([[2.5]])

    z = torch.randn(1, 1)

    delta = model(x, z)

    print("\nInput x")
    print(x)

    print("\nRandom z")
    print(z)

    print("\nPredicted Δx")
    print(delta)

    print("\nPredicted next position")

    print(x + delta)


# "Every time you rerun it,

# z

# changes,

# so

# Δx

# changes too."