import random
import chess
from .base import Bot


class RandomBot(Bot):
    async def choose_move(self, board: chess.Board) -> chess.Move:
        return random.choice(list(board.legal_moves))
