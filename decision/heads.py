import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, depth=2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DecisionHeads(nn.Module):
    """Auxiliary control heads operating on latent embeddings."""

    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.reward = MLPHead(latent_dim + action_dim, 1)
        self.value = MLPHead(latent_dim, 1)
        self.action = MLPHead(latent_dim * 2, action_dim)

    def predict_reward(self, z, a):
        return self.reward(torch.cat([z, a], dim=-1)).squeeze(-1)

    def predict_value(self, z):
        return self.value(z).squeeze(-1)

    def inverse_action(self, z_t, z_tp1):
        return self.action(torch.cat([z_t, z_tp1], dim=-1))
