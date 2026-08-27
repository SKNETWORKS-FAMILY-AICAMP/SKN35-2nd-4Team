"""E 담당 선수 추천 Autoencoder 네트워크.

모듈 최상단에 정의해 torch 저장 시 import 가능한 클래스 경로를 유지한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PlayerAutoencoderNet(nn.Module):
    """표준화된 선수 전력 피처를 낮은 차원의 잠재 벡터로 압축한다."""

    def __init__(self, n_features: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encode(x))
