import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossFitnessRouter(nn.Module):
    def __init__(self, input_dim: int, num_features: int = 64):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_features),
            nn.Sigmoid(),
        )
        self.feature_amplifier = nn.Parameter(torch.ones(num_features))
        self.feature_suppressor = nn.Parameter(torch.zeros(num_features))

    def forward(self, sparse_features):
        gate = self.gating(sparse_features)
        modulated = sparse_features * gate * self.feature_amplifier + \
                    sparse_features * (1 - gate) * self.feature_suppressor
        return modulated, gate


class PathogenicityPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.router = CrossFitnessRouter(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, sparse_features):
        modulated, gate = self.router(sparse_features)
        logits = self.classifier(modulated)
        return logits, gate
