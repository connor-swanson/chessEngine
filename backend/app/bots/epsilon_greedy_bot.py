from __future__ import annotations

import random
from typing import Optional

import chess

from .base import Bot


class EpsilonGreedyBot(Bot):
    """Wraps any bot and plays a uniformly random legal move with probability epsilon.

    This breaks the symmetry of bot-vs-bot self-play and produces more diverse
    games.  Examples: heuristic-e10 (strong, occasional slip), ab2-e25 (depth-2
    alpha-beta that plays randomly 25% of the time).
    """

    def __init__(self, base_bot: Bot, epsilon: float, seed: Optional[int] = None):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.base_bot = base_bot
        self.epsilon = epsilon
        self._rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        if self._rng.random() < self.epsilon:
            return self._rng.choice(list(board.legal_moves))
        return self.base_bot.select_move(board)

    async def choose_move(self, board: chess.Board) -> chess.Move:
        if self._rng.random() < self.epsilon:
            return self._rng.choice(list(board.legal_moves))
        return await self.base_bot.choose_move(board)
