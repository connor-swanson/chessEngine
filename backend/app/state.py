"""Builds the GameStateResponse dict for a game, merging board state with endings
that happen off the board (resignation, loss on time)."""
from __future__ import annotations

from typing import Optional, Tuple

import chess

from .chess_utils import board_outcome, board_to_state
from .game_manager import game_manager
from .models import Game


def game_ending(board: chess.Board, game: Game) -> Optional[Tuple[str, str]]:
    ending = game_manager.get_ending(game.id)
    if ending is not None:
        return ending
    outcome = board_outcome(board)
    if outcome is not None:
        return outcome
    if game.result:  # ended off the board before a server restart
        return game.result, "resignation"
    return None


def game_state(board: chess.Board, game: Game) -> dict:
    return {
        **board_to_state(board, game.id, game_ending(board, game)),
        "white_player": game.white_player,
        "black_player": game.black_player,
    }
