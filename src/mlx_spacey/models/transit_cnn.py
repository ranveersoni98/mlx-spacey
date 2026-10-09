"""MLX transit classifier: multi-view 1D CNN + stellar branch."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class ViewEncoder(nn.Module):
    """Small 1D CNN: (B, L, 1) -> (B, out_dim)."""

    def __init__(self, out_dim: int = 64, hidden: int = 32, p_drop: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden * 2, hidden * 2, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden * 2)
        self.norm3 = nn.LayerNorm(hidden * 2)
        self.drop = nn.Dropout(p_drop)
        self.proj = nn.Linear(hidden * 2, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, L) -> (B, L, 1)
        if len(x.shape) == 2:
            x = mx.expand_dims(x, -1)
        h = self.conv1(x)
        h = nn.relu(self.norm1(h))
        h = h[:, ::2, :]  # (B, L/2, C)
        h = self.conv2(h)
        h = nn.relu(self.norm2(h))
        h = h[:, ::2, :]  # (B, L/4, C)
        h = self.conv3(h)
        h = nn.relu(self.norm3(h))
        h = mx.mean(h, axis=1)  # global average pool -> (B, C)
        h = self.drop(h)
        return self.proj(h)


class TransitClassifier(nn.Module):
    """Global (201) + local (81) + stellar features -> 3 logits."""

    def __init__(
        self,
        num_classes: int = 3,
        stellar_dim: int = 1,
        enc_dim: int = 64,
        mlp_dim: int = 128,
        p_drop: float = 0.3,
    ):
        super().__init__()
        self.global_enc = ViewEncoder(out_dim=enc_dim)
        self.local_enc = ViewEncoder(out_dim=enc_dim)
        self.stellar_mlp = nn.Sequential(
            nn.Linear(stellar_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )
        self.head = nn.Sequential(
            nn.Linear(enc_dim * 2 + 16, mlp_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(mlp_dim, num_classes),
        )

    def __call__(self, g: mx.array, l: mx.array, s: mx.array) -> mx.array:
        hg = self.global_enc(g)
        hl = self.local_enc(l)
        hs = self.stellar_mlp(s)
        h = mx.concatenate([hg, hl, hs], axis=-1)
        return self.head(h)


def create_model(**kwargs) -> TransitClassifier:
    return TransitClassifier(**kwargs)


def count_params(model: nn.Module) -> int:
    from mlx.utils import tree_flatten

    return sum(v.size for _, v in tree_flatten(model.parameters()))
