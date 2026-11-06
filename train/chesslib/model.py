"""The two-headed policy/value network used by every notebook.

Architecture (AlphaZero-style, scaled down for a laptop)::

    planes (21, 8, 8)
      -> conv 3x3 stem
      -> N residual blocks (conv-BN-ReLU-conv-BN + skip)
      ├─> policy head: every square becomes a token; logits for "from-square f
      │   to to-square t" are the dot product query(f) · key(t)  (64x64 = 4096),
      │   plus a small linear head for the 72 under-promotion moves.
      └─> value head: win / draw / loss logits from the side-to-move's view.

Why a dot-product policy head?  A plain ``Linear(…, 4096)`` head needs millions
of weights and has to learn separately that "knight on g1 to f3" and "knight
on b1 to c3" are similar.  Scoring (from, to) pairs with per-square embeddings
is ~100x smaller and shares that structure (Leela Chess Zero uses the same idea).

Why win/draw/loss instead of one tanh value?  With many drawn games a scalar
can't tell "certain draw" from "coin flip"; three probabilities can, and the
scalar is still available as ``P(win) - P(loss)``.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import ACTION_SIZE, NUM_PLANES


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class ChessNet(nn.Module):
    def __init__(self, blocks: int = 6, filters: int = 64, policy_dim: int = 32):
        super().__init__()
        self.config = {"blocks": blocks, "filters": filters, "policy_dim": policy_dim}
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters), nn.ReLU(inplace=True))
        self.tower = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])
        # policy
        self.query = nn.Linear(filters, policy_dim)
        self.key = nn.Linear(filters, policy_dim)
        self.promo = nn.Sequential(
            nn.Conv2d(filters, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Flatten(), nn.Linear(8 * 64, 72))
        self.policy_scale = 1.0 / math.sqrt(policy_dim)
        # value
        self.value = nn.Sequential(
            nn.Conv2d(filters, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Flatten(), nn.Linear(8 * 64, 128), nn.ReLU(inplace=True), nn.Linear(128, 3))

    def forward(self, x):
        """x: (B, 21, 8, 8) float -> (policy_logits (B, 4168), wdl_logits (B, 3))."""
        f = self.tower(self.stem(x))
        tokens = f.flatten(2).transpose(1, 2)                       # (B, 64, C), token = square
        q, k = self.query(tokens), self.key(tokens)
        move_logits = torch.bmm(q, k.transpose(1, 2)) * self.policy_scale  # (B, 64 from, 64 to)
        policy = torch.cat([move_logits.flatten(1), self.promo(f)], dim=1)
        return policy, self.value(f)


def wdl_to_scalar(wdl_logits: torch.Tensor) -> torch.Tensor:
    """Expected score in [-1, 1] for the side to move: P(win) - P(loss)."""
    p = torch.softmax(wdl_logits, dim=-1)
    return p[..., 0] - p[..., 2]


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log-softmax over legal moves only (mask: bool, True = legal).

    Illegal entries come out as a large negative number (not -inf), which keeps
    ``probs * log_probs`` finite on every backend (MPS included).
    """
    return torch.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path, model: ChessNet, **metadata) -> None:
    torch.save({"model_state_dict": model.state_dict(), "model_config": model.config,
                "metadata": metadata}, path)


def load_checkpoint(path, device="cpu") -> ChessNet:
    """Load a checkpoint written by :func:`save_checkpoint` (model in eval mode).

    The metadata dict is attached as ``model.metadata``.
    """
    ckpt = torch.load(path, map_location=device)
    model = ChessNet(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.metadata = ckpt.get("metadata", {})
    return model.eval()
