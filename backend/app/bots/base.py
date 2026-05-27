from abc import ABC, abstractmethod
import chess


class Bot(ABC):
    @abstractmethod
    async def choose_move(self, board: chess.Board) -> chess.Move:
        ...
