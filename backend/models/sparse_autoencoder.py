import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, sparsity_lambda: float = 1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_lambda = sparsity_lambda
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight, a=0.01)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=0.01)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        encoded = self.encoder(x)
        sparse = F.relu(encoded)
        reconstructed = self.decoder(sparse)
        return reconstructed, sparse

    def encode(self, x):
        return F.relu(self.encoder(x))

    def sparsity_loss(self, sparse_activations):
        mean_activation = sparse_activations.mean(dim=0)
        kl = mean_activation * torch.log(mean_activation + 1e-10) + \
             (1 - mean_activation) * torch.log(1 - mean_activation + 1e-10)
        return self.sparsity_lambda * kl.sum()
