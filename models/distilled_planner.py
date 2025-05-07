import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

class DistilledPlannerModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class DistilledPlanner:
    def __init__(self, input_dim: int = 16, output_dim: int = 4, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DistilledPlannerModel(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 50, val_split: float = 0.2) -> None:
        # Prepare data
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)
        # Split
        n = len(X)
        idx = int(n * (1 - val_split))
        X_train, Y_train = X[:idx], Y[:idx]
        X_val, Y_val = X[idx:], Y[idx:]
        for epoch in range(1, epochs+1):
            self.model.train()
            self.optimizer.zero_grad()
            pred = self.model(X_train)
            loss = self.criterion(pred, Y_train)
            loss.backward()
            self.optimizer.step()
            if X_val.size(0) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val)
                    val_loss = self.criterion(val_pred, Y_val)
                print(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")
            else:
                print(f"Epoch {epoch}: train_loss={loss.item():.4f}")

    def predict(self, z_t: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.tensor(z_t, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.model(x).cpu().numpy().squeeze(0)
        return y

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))