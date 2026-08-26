"""StrengthLSTM 의 torch.nn.Module 정의 — 별도 파일로 분리한 이유 2가지.

1. 피클링 — torch.save() 는 내부적으로 pickle 을 쓰는데, 함수/메서드 안에 정의된
   클래스(closure)는 import 경로가 없어 저장에 실패한다. 모듈 최상단에 있어야 한다.
2. 임포트 순서 — torch 를 먼저 import 한 상태에서 xgboost.fit() 을 호출하면
   macOS(arm64)에서 두 라이브러리의 OpenMP/Accelerate 초기화가 충돌해 세그폴트가 난다.
   이 파일을 strength_ts.py 최상단이 아니라 StrengthLSTM._fit()/_predict() 안에서만
   지연 임포트해야, StrengthXGB 가 먼저 xgboost 를 초기화한 뒤에 torch 가 들어와 안전하다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedLSTMNet(nn.Module):
    """LSTMCell 을 타임스텝별로 직접 돌려 패딩(pad_value=0.0)을 무시한다.

    build_sequences() 는 앞쪽만 0 으로 패딩하므로(마지막 스텝은 항상 실측치),
    패딩된 스텝에서는 은닉 상태 갱신을 건너뛰어 Keras Masking 과 동일하게 동작시킨다.
    """

    def __init__(self, n_feat: int, units: int, dropout: float):
        super().__init__()
        self.cell = nn.LSTMCell(n_feat, units)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(units, 16), nn.ReLU(), nn.Linear(16, 1))
        self.units = units

    def forward(self, x):
        # x: (batch, seq_len, n_feat). 한 스텝의 모든 피처가 0이면 패딩으로 본다
        mask = x.abs().sum(dim=-1) > 1e-8   # (batch, seq_len)
        b = x.size(0)
        h = x.new_zeros(b, self.units)
        c = x.new_zeros(b, self.units)
        for t in range(x.size(1)):
            h_new, c_new = self.cell(x[:, t, :], (h, c))
            m = mask[:, t].unsqueeze(1)
            h = torch.where(m, h_new, h)
            c = torch.where(m, c_new, c)
        return self.head(self.dropout(h)).squeeze(-1)
