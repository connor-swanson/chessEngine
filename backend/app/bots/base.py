from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import chess


class Bot(ABC):
    """Chess bot interface.

    `select_move` is the plain synchronous implementation (used by the parallel
    game generator and tests); `choose_move` is the async entry point used by the
    FastAPI app.  Neither may mutate the caller's board.
    """

    def select_move(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError

    @abstractmethod
    async def choose_move(self, board: chess.Board) -> chess.Move:
        ...


class SyncBot(Bot):
    """Bot whose async method just runs `select_move` in an executor."""

    async def choose_move(self, board: chess.Board) -> chess.Move:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.select_move, board.copy())
