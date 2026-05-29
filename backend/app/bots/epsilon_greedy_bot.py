from __future__ import annotations

import random
import chess

from .base import Bot


class EpsilonGreedyBot(Bot):
    """Wraps any bot and plays a uniformly random legal move with probability epsilon.

    This breaks the symmetry of bot-vs-bot self-play and produces far more decisive
    games than two identical deterministic bots.  Useful for generating training data
    with a natural skill gradient:

        heuristic-e10  — strong, occasional slip   (epsilon=0.10)
        heuristic-e20  — medium, noticeable errors (epsilon=0.20)
        heuristic-e35  — weak, frequent mistakes   (epsilon=0.35)

    Matchup ideas for diverse training data:
        heuristic-e20  vs  heuristic-e20   — balanced but decisive
        heuristic      vs  heuristic-e35   — asymmetric skill
        random         vs  heuristic       — lots of decisive wins for heuristic
    """

    def __init__(self, base_bot: Bot, epsilon: float):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.base_bot = base_bot
        self.epsilon = epsilon

    async def choose_move(self, board: chess.Board) -> chess.Move:
        if random.random() < self.epsilon:
            return random.choice(list(board.legal_moves))
        return await self.base_bot.choose_move(board)
