from __future__ import annotations

import random
from typing import Optional

import chess

from .base import Bot


class RandomBot(Bot):
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        return self._rng.choice(list(board.legal_moves))

    async def choose_move(self, board: chess.Board) -> chess.Move:
        return self.select_move(board)
