from __future__ import annotations
import chess
import chess.engine
from .base import Bot

STOCKFISH_PATH = "stockfish"


class StockfishBot(Bot):
    """Spawns the stockfish binary lazily (per move), so constructing this
    never fails even if stockfish is not installed."""

    def __init__(self, think_time: float = 0.1, skill_level: int = 20):
        self.think_time = think_time
        self.skill_level = skill_level

    def select_move(self, board: chess.Board) -> chess.Move:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        try:
            engine.configure({"Skill Level": self.skill_level})
            return engine.play(board, chess.engine.Limit(time=self.think_time)).move
        finally:
            engine.quit()

    async def choose_move(self, board: chess.Board) -> chess.Move:
        transport, engine = await chess.engine.popen_uci(STOCKFISH_PATH)
        try:
            await engine.configure({"Skill Level": self.skill_level})
            result = await engine.play(board, chess.engine.Limit(time=self.think_time))
        finally:
            await engine.quit()
        return result.move
