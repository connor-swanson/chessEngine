from __future__ import annotations
from typing import Optional
from .base import Bot
from .random_bot import RandomBot
from .heuristic_bot import HeuristicBot
from .stockfish_bot import StockfishBot

BOT_REGISTRY: dict[str, Bot] = {
    "random": RandomBot(),
    "heuristic": HeuristicBot(),
    "stockfish": StockfishBot(think_time=0.1, skill_level=20),
    "stockfish-medium": StockfishBot(think_time=0.1, skill_level=12),
    "stockfish-easy": StockfishBot(think_time=0.1, skill_level=5),
}


def get_bot(player_type: str) -> Optional[Bot]:
    return BOT_REGISTRY.get(player_type)
