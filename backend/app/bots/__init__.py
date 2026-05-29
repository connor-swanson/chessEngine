from __future__ import annotations
from typing import Optional
from .base import Bot
from .random_bot import RandomBot
from .heuristic_bot import HeuristicBot
from .stockfish_bot import StockfishBot
from .epsilon_greedy_bot import EpsilonGreedyBot

_heuristic = HeuristicBot()

BOT_REGISTRY: dict[str, Bot] = {
    "random":           RandomBot(),
    "heuristic":        _heuristic,
    # Epsilon-greedy variants — HeuristicBot that occasionally plays randomly.
    # Use these for generating decisive, diverse training data.
    "heuristic-e10":    EpsilonGreedyBot(_heuristic, epsilon=0.10),
    "heuristic-e20":    EpsilonGreedyBot(_heuristic, epsilon=0.20),
    "heuristic-e35":    EpsilonGreedyBot(_heuristic, epsilon=0.35),
    "stockfish":        StockfishBot(think_time=0.1, skill_level=20),
    "stockfish-medium": StockfishBot(think_time=0.1, skill_level=12),
    "stockfish-easy":   StockfishBot(think_time=0.1, skill_level=5),
}


def get_bot(player_type: str) -> Optional[Bot]:
    return BOT_REGISTRY.get(player_type)
