import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Exit Network
# ============================================================

class ExitNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Load Dataset
# ============================================================

def load_dataset():
    data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "code_1d",
        "artifacts_bd",
        "data_exit.npz",
    )

    data = np.load(data_path)

    x = data["x"].astype(np.float32)
    y = data["exited"].astype(np.float32)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return x, y


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # ----------------------------
    # Load data
    # ----------------------------
    x, y = load_dataset()

    X = torch.tensor(x, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X, Y)

    dataloader = DataLoader(
        dataset,
        batch_size=2048,
        shuffle=True
    )

    # ----------------------------
    # Build model
    # ----------------------------
    model = ExitNet()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.003
    )

    loss_fn = nn.BCELoss()

    # ----------------------------
    # Train
    # ----------------------------
    epochs = 10

    for epoch in range(epochs):

        total_loss = 0.0

        model.train()

        for batch_x, batch_y in dataloader:

            prediction = model(batch_x)

            loss = loss_fn(prediction, batch_y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch+1:02d} | Loss = {avg_loss:.6f}")

    # ----------------------------
    # Test the network
    # ----------------------------
    model.eval()

    test_points = torch.tensor(
        [
            [0.2],
            [1.5],
            [3.0],
            [5.0],
            [5.8],
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        probs = model(test_points)

    print("\nPredicted exit probabilities\n")

    for x_value, p in zip(test_points, probs):
        print(
            f"x = {x_value.item():.2f}  -->  P(exit) = {p.item():.4f}"
        )