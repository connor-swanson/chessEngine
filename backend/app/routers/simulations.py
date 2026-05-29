from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import chess
import chess.pgn
from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..database import SessionLocal
from ..models import Game, Position
from ..bots import get_bot
from ..schemas import RunSimulationRequest, SimulationStatusResponse

router = APIRouter(prefix="/simulations", tags=["simulations"])

_CONCURRENCY = 10


@dataclass
class SimulationState:
    id: int
    total: int
    white_player: str
    black_player: str
    completed: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


_simulations: dict[int, SimulationState] = {}
_sim_counter = 0
_counter_lock = asyncio.Lock()


def _board_to_pgn(board: chess.Board) -> str:
    game = chess.pgn.Game.from_board(board)
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return game.accept(exporter)


async def _run_single_game(white_player: str, black_player: str) -> str:
    board = chess.Board()

    async with SessionLocal() as db:
        game = Game(white_player=white_player, black_player=black_player, source="simulation")
        db.add(game)
        await db.commit()
        await db.refresh(game)

        db.add(Position(game_id=game.id, move_number=0, fen=board.fen(), move_uci=None, is_terminal=False))
        await db.commit()

        while not board.is_game_over():
            player_type = white_player if board.turn == chess.WHITE else black_player
            bot = get_bot(player_type)
            if not bot:
                raise ValueError(f"Unknown player type: {player_type}")
            move = await bot.choose_move(board)
            move_uci = move.uci()
            board.push(move)

            is_terminal = board.is_game_over()
            db.add(Position(
                game_id=game.id,
                move_number=len(board.move_stack),
                fen=board.fen(),
                move_uci=move_uci,
                is_terminal=is_terminal,
            ))

        result = board.result()
        game.result = result
        game.pgn = _board_to_pgn(board)
        await db.commit()

    return result


async def _run_simulation(sim_id: int):
    sim = _simulations[sim_id]
    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async def run_one():
        async with semaphore:
            try:
                result = await _run_single_game(sim.white_player, sim.black_player)
                if result == "1-0":
                    sim.white_wins += 1
                elif result == "0-1":
                    sim.black_wins += 1
                else:
                    sim.draws += 1
            except Exception:
                pass
            finally:
                sim.completed += 1

    await asyncio.gather(*[run_one() for _ in range(sim.total)])
    sim.status = "done"
    sim.finished_at = time.time()


@router.post("", response_model=SimulationStatusResponse)
async def start_simulation(req: RunSimulationRequest, background_tasks: BackgroundTasks):
    global _sim_counter
    async with _counter_lock:
        _sim_counter += 1
        sim_id = _sim_counter

    sim = SimulationState(
        id=sim_id,
        total=req.num_games,
        white_player=req.white_player,
        black_player=req.black_player,
    )
    _simulations[sim_id] = sim
    background_tasks.add_task(_run_simulation, sim_id)

    return SimulationStatusResponse(
        id=sim.id,
        status=sim.status,
        total=sim.total,
        completed=sim.completed,
        white_wins=sim.white_wins,
        black_wins=sim.black_wins,
        draws=sim.draws,
        white_player=sim.white_player,
        black_player=sim.black_player,
    )


@router.get("/{sim_id}", response_model=SimulationStatusResponse)
async def get_simulation(sim_id: int):
    sim = _simulations.get(sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return SimulationStatusResponse(
        id=sim.id,
        status=sim.status,
        total=sim.total,
        completed=sim.completed,
        white_wins=sim.white_wins,
        black_wins=sim.black_wins,
        draws=sim.draws,
        white_player=sim.white_player,
        black_player=sim.black_player,
    )
