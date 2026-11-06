from __future__ import annotations
from typing import Callable, Dict, Optional
from .base import Bot
from .random_bot import RandomBot
from .heuristic_bot import HeuristicBot
from .alphabeta_bot import AlphaBetaBot
from .stockfish_bot import StockfishBot
from .epsilon_greedy_bot import EpsilonGreedyBot
from .neural_bot import NeuralBot, availability, checkpoint_path

# name -> factory(seed) -> fresh Bot.  Use these (not BOT_REGISTRY) when you need
# independent RNG state, e.g. one bot per worker process / per game.
BOT_FACTORIES: Dict[str, Callable[[Optional[int]], Bot]] = {
    "random":           lambda seed=None: RandomBot(seed),
    "heuristic":        lambda seed=None: HeuristicBot(seed),
    # Epsilon-greedy heuristic variants: occasionally play a random move.
    "heuristic-e10":    lambda seed=None: EpsilonGreedyBot(HeuristicBot(seed), 0.10, seed),
    "heuristic-e20":    lambda seed=None: EpsilonGreedyBot(HeuristicBot(seed), 0.20, seed),
    "heuristic-e35":    lambda seed=None: EpsilonGreedyBot(HeuristicBot(seed), 0.35, seed),
    # Alpha-beta (material + PST + quiescence), random among moves within 10cp.
    "ab1":              lambda seed=None: AlphaBetaBot(1, seed, noise_cp=10),
    "ab2":              lambda seed=None: AlphaBetaBot(2, seed, noise_cp=10),
    "ab3":              lambda seed=None: AlphaBetaBot(3, seed, noise_cp=10),
    "ab2-e10":          lambda seed=None: EpsilonGreedyBot(AlphaBetaBot(2, seed, noise_cp=10), 0.10, seed),
    "ab2-e25":          lambda seed=None: EpsilonGreedyBot(AlphaBetaBot(2, seed, noise_cp=10), 0.25, seed),
    "stockfish":        lambda seed=None: StockfishBot(think_time=0.1, skill_level=20),
    "stockfish-medium": lambda seed=None: StockfishBot(think_time=0.1, skill_level=12),
    "stockfish-easy":   lambda seed=None: StockfishBot(think_time=0.1, skill_level=5),
    # Trained neural engine (needs torch + train/checkpoints/*.pt; errors when used if missing).
    "engine":           lambda seed=None: NeuralBot(checkpoint_path("engine.pt"), sims=200),
    "engine-fast":      lambda seed=None: NeuralBot(checkpoint_path("engine.pt"), sims=50),
    "engine-policy":    lambda seed=None: NeuralBot(checkpoint_path("engine.pt"), sims=0),
    "sft":              lambda seed=None: NeuralBot(checkpoint_path("sft.pt"), sims=100),
    "mcts":             lambda seed=None: NeuralBot(checkpoint_path("mcts.pt"), sims=100),
}


def make_bot(name: str, seed: Optional[int] = None) -> Bot:
    return BOT_FACTORIES[name](seed)


BOT_REGISTRY: Dict[str, Bot] = {name: f(None) for name, f in BOT_FACTORIES.items()}


def get_bot(player_type: str) -> Optional[Bot]:
    return BOT_REGISTRY.get(player_type)


def available_bots() -> Dict[str, Dict[str, object]]:
    """name -> {available, reason}. Neural bots are unavailable without torch / their checkpoint."""
    out: Dict[str, Dict[str, object]] = {}
    for name, bot in BOT_REGISTRY.items():
        if isinstance(bot, NeuralBot):
            ok, reason = availability(bot.checkpoint)
        else:
            ok, reason = True, None
        out[name] = {"available": ok, "reason": reason}
    return out
