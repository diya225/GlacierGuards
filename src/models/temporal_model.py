"""Temporal anomaly models: predict the next lake extent from N past masks.

Two interchangeable heads (pipeline.yaml model.temporal.type):
  conv_lstm : ConvLSTM encoder over (mask, dt) frames -> 1x1 conv -> next-mask logits
  attention : per-frame CNN encoder, temporal self-attention weighted by target dt, decoder
Input  x : (B, N, 1, H, W) past mask probabilities
Input dt : (B, N) days from each past frame to the target date (decreasing; handles irregular revisit)
Output   : (B, 1, H, W) logits of the expected next extent; residual = observed - sigmoid(output)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, hid, k=3):
        super().__init__()
        self.hid = hid
        self.gates = nn.Conv2d(cin + hid, 4 * hid, k, padding=k // 2)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.chunk(self.gates(torch.cat([x, h], 1)), 4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        # frame channels: mask, dt-since-previous (scaled), dt-to-target (scaled)
        self.cells = nn.ModuleList([ConvLSTMCell(3 if i == 0 else hidden, hidden) for i in range(layers)])
        self.head = nn.Conv2d(hidden, 1, 1)
        self.hidden = hidden

    def forward(self, x, dt):
        B, N, _, H, W = x.shape
        states = [(x.new_zeros(B, self.hidden, H, W), x.new_zeros(B, self.hidden, H, W)) for _ in self.cells]
        for t in range(N):
            prev = (dt[:, t - 1] - dt[:, t]) if t else torch.zeros_like(dt[:, 0])
            inp = torch.cat([x[:, t], (prev / 30.0)[:, None, None, None].expand(B, 1, H, W),
                             (dt[:, t] / 30.0)[:, None, None, None].expand(B, 1, H, W)], 1)
            for li, cell in enumerate(self.cells):
                states[li] = cell(inp, states[li])
                inp = states[li][0]
        return self.head(states[-1][0]) + _logit(x[:, -1])  # residual connection: persistence prior


class AttentionForecaster(nn.Module):
    def __init__(self, hidden=64, layers=2, heads=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.att = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.q = nn.Parameter(torch.zeros(1, 1, hidden))
        self.k = nn.Linear(1, hidden)
        self.dec = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, 1, 1))

    def forward(self, x, dt):
        B, N, _, H, W = x.shape
        f = self.enc(x.flatten(0, 1)).view(B, N, -1, H, W)  # B,N,C,H,W
        tok = f.mean((3, 4)) + self.k(dt[:, :, None] / 30.0)
        q = self.q.expand(B, 1, -1)
        _, w = self.att(q, tok, tok)                        # B,1,N attention weights over time
        fused = (f * w[:, 0, :, None, None, None]).sum(1)
        return self.dec(fused) + _logit(x[:, -1])


def _logit(p, eps=1e-4):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


def build_temporal(kind="conv_lstm", hidden=64, layers=2):
    return ConvLSTMForecaster(hidden, layers) if kind == "conv_lstm" else AttentionForecaster(hidden, layers)
