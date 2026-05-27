from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class CreateGameRequest(BaseModel):
    white_player: str = "human"
    black_player: str = "random"


class MoveRequest(BaseModel):
    move_uci: str


class GameStateResponse(BaseModel):
    id: int
    white_player: str
    black_player: str
    fen: str
    legal_moves: list[str]
    turn: str
    result: Optional[str]
    move_history: list[str]
    is_game_over: bool


class PositionResponse(BaseModel):
    id: int
    move_number: int
    fen: str
    move_uci: Optional[str]
    is_terminal: bool
    tensor_shape: list[int]
    tensor: list


class GameSummary(BaseModel):
    id: int
    white_player: str
    black_player: str
    result: Optional[str]
    created_at: datetime
    move_count: int


class RunSimulationRequest(BaseModel):
    num_games: int
    white_player: str = "random"
    black_player: str = "random"


class SimulationStatusResponse(BaseModel):
    id: int
    status: str
    total: int
    completed: int
    white_wins: int
    black_wins: int
    draws: int
    white_player: str
    black_player: str
